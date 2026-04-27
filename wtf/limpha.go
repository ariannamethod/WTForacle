package wtf

// limpha.go — WTForacle's persistent memory.
//
// Port of limpha/memory.py to Go. Single SQLite file at ~/.wtforacle/limpha.db,
// FTS5 over (prompt, response) for /recall, BM25 ranking. Auto-stores every
// turn — no /save command. Quality score rewards reddit slang, penalizes
// generic-assistant phrasing.
//
// Adapted from yent's limpha subsystem.

import (
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

const limphaSchema = `
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    session_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    temperature REAL DEFAULT 0.9,
    quality REAL DEFAULT 0.5,
    access_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON conversations(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conv_quality ON conversations(quality DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
    prompt,
    response,
    content=conversations,
    content_rowid=id,
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS conv_fts_insert AFTER INSERT ON conversations BEGIN
    INSERT INTO conversations_fts(rowid, prompt, response)
    VALUES (new.id, new.prompt, new.response);
END;

CREATE TRIGGER IF NOT EXISTS conv_fts_delete AFTER DELETE ON conversations BEGIN
    INSERT INTO conversations_fts(conversations_fts, rowid, prompt, response)
    VALUES ('delete', old.id, old.prompt, old.response);
END;

CREATE TRIGGER IF NOT EXISTS conv_fts_update AFTER UPDATE ON conversations BEGIN
    INSERT INTO conversations_fts(conversations_fts, rowid, prompt, response)
    VALUES ('delete', old.id, old.prompt, old.response);
    INSERT INTO conversations_fts(rowid, prompt, response)
    VALUES (new.id, new.prompt, new.response);
END;

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    last_active REAL NOT NULL,
    turn_count INTEGER DEFAULT 0,
    avg_quality REAL DEFAULT 0.0
);
`

// Conversation is one stored turn of dialogue.
type Conversation struct {
	ID          int64
	Prompt      string
	Response    string
	Quality     float64
	Temperature float64
}

// Stats summarises the database.
type Stats struct {
	TotalConversations int
	TotalSessions      int
	AvgQuality         float64
	DBPath             string
	DBSizeBytes        int64
}

// Limpha is the SQLite-backed memory subsystem. All methods are safe to call
// from a single goroutine; the REPL is single-threaded.
type Limpha struct {
	db        *sql.DB
	sessionID string
	dbPath    string
}

// OpenLimpha opens (and initializes) ~/.wtforacle/limpha.db.
func OpenLimpha() (*Limpha, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}
	dir := filepath.Join(home, ".wtforacle")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	path := filepath.Join(dir, "limpha.db")

	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	if _, err := db.Exec("PRAGMA journal_mode=WAL"); err != nil {
		return nil, err
	}
	if _, err := db.Exec("PRAGMA synchronous=NORMAL"); err != nil {
		return nil, err
	}
	if _, err := db.Exec(limphaSchema); err != nil {
		return nil, fmt.Errorf("schema: %w", err)
	}

	sid := newSessionID()
	now := float64(time.Now().UnixNano()) / 1e9
	if _, err := db.Exec(
		"INSERT OR IGNORE INTO sessions (session_id, started_at, last_active) VALUES (?, ?, ?)",
		sid, now, now,
	); err != nil {
		return nil, err
	}

	return &Limpha{db: db, sessionID: sid, dbPath: path}, nil
}

// Close releases the database handle.
func (l *Limpha) Close() error {
	if l.db == nil {
		return nil
	}
	return l.db.Close()
}

// Store records a single turn. Quality is computed from the response text.
func (l *Limpha) Store(prompt, response string, temperature float64) (int64, error) {
	now := float64(time.Now().UnixNano()) / 1e9
	q := computeQuality(prompt, response)

	res, err := l.db.Exec(
		`INSERT INTO conversations
		 (timestamp, session_id, prompt, response, temperature, quality)
		 VALUES (?, ?, ?, ?, ?, ?)`,
		now, l.sessionID, prompt, response, temperature, q,
	)
	if err != nil {
		return 0, err
	}
	id, _ := res.LastInsertId()

	if _, err := l.db.Exec(
		`UPDATE sessions SET
		   last_active = ?,
		   avg_quality = (avg_quality * turn_count + ?) / (turn_count + 1),
		   turn_count = turn_count + 1
		 WHERE session_id = ?`,
		now, q, l.sessionID,
	); err != nil {
		return id, err
	}
	return id, nil
}

// Search runs an FTS5 query (BM25 ranked) and returns up to `limit` hits.
func (l *Limpha) Search(query string, limit int) ([]Conversation, error) {
	if strings.TrimSpace(query) == "" {
		return nil, nil
	}
	rows, err := l.db.Query(
		`SELECT c.id, c.prompt, c.response, c.quality, c.temperature
		 FROM conversations_fts fts
		 JOIN conversations c ON c.id = fts.rowid
		 WHERE conversations_fts MATCH ?
		 ORDER BY bm25(conversations_fts)
		 LIMIT ?`,
		query, limit,
	)
	if err != nil {
		// FTS5 syntax error → empty result, not a hard fail.
		return nil, nil
	}
	defer rows.Close()
	return scanConversations(rows)
}

// Recent returns up to `limit` most-recent turns from THIS session,
// ordered chronologically (oldest first).
func (l *Limpha) Recent(limit int) ([]Conversation, error) {
	rows, err := l.db.Query(
		`SELECT id, prompt, response, quality, temperature FROM conversations
		 WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?`,
		l.sessionID, limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	convs, err := scanConversations(rows)
	if err != nil {
		return nil, err
	}
	// Reverse to chronological order.
	for i, j := 0, len(convs)-1; i < j; i, j = i+1, j-1 {
		convs[i], convs[j] = convs[j], convs[i]
	}
	return convs, nil
}

// Stats summarises the whole database (not just the current session).
func (l *Limpha) Stats() (Stats, error) {
	var st Stats
	st.DBPath = l.dbPath

	if err := l.db.QueryRow("SELECT COUNT(*) FROM conversations").Scan(&st.TotalConversations); err != nil {
		return st, err
	}
	if err := l.db.QueryRow("SELECT COUNT(*) FROM sessions").Scan(&st.TotalSessions); err != nil {
		return st, err
	}
	var avg sql.NullFloat64
	if err := l.db.QueryRow("SELECT AVG(quality) FROM conversations").Scan(&avg); err != nil {
		return st, err
	}
	if avg.Valid {
		st.AvgQuality = avg.Float64
	}
	if info, err := os.Stat(l.dbPath); err == nil {
		st.DBSizeBytes = info.Size()
	}
	return st, nil
}

func scanConversations(rows *sql.Rows) ([]Conversation, error) {
	var out []Conversation
	for rows.Next() {
		var c Conversation
		if err := rows.Scan(&c.ID, &c.Prompt, &c.Response, &c.Quality, &c.Temperature); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

func newSessionID() string {
	b := make([]byte, 4)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("s%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)
}

// computeQuality mirrors limpha/memory.py:_compute_quality. Same constants,
// same blend — keeps existing limpha.db files comparable across Python and
// Go writers.
func computeQuality(prompt, response string) float64 {
	resp := strings.TrimSpace(response)
	if resp == "" {
		return 0.0
	}
	respLen := len(resp)
	promptLen := len(strings.TrimSpace(prompt))
	if promptLen < 1 {
		promptLen = 1
	}

	var lengthScore float64
	switch {
	case respLen < 10:
		lengthScore = 0.1
	case respLen < 50:
		lengthScore = 0.3
	case respLen < 200:
		lengthScore = 0.5 + 0.3*float64(respLen-50)/150
	default:
		lengthScore = 0.8
	}

	ratio := float64(respLen) / float64(promptLen)
	var ratioScore float64
	switch {
	case ratio < 0.3:
		ratioScore = 0.2
	case ratio > 10:
		ratioScore = 0.6
	default:
		ratioScore = 0.7
	}

	respLower := strings.ToLower(response)
	slang := []string{"bro", "tbh", "ngl", "imo", "lmao", "lol", "bruh",
		"nah", "fr", "literally", "actually", "honestly"}
	slangCount := 0
	for _, s := range slang {
		slangCount += strings.Count(respLower, s)
	}
	cynicism := math.Min(float64(slangCount)*0.03, 0.15)

	boring := []string{"as an ai", "i cannot", "i apologize", "great question"}
	boringCount := 0
	for _, b := range boring {
		if strings.Contains(respLower, b) {
			boringCount++
		}
	}
	boringPenalty := float64(boringCount) * 0.1

	q := 0.6*lengthScore + 0.4*ratioScore + cynicism - boringPenalty
	if q < 0 {
		q = 0
	} else if q > 1 {
		q = 1
	}
	return q
}
