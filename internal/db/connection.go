package db

import (
	"context"
	"database/sql"
	_ "embed"

	"github.com/cozy-creator/gen-server/internal/config"

	_ "github.com/mattn/go-sqlite3"
)

//go:embed sql/schema.sql
var schema string

func NewConnection(config *config.Config) (*Queries, error) {
	driver := config.DB.Driver
	dsn := config.DB.DSN

	db, err := sql.Open(driver, dsn)
	if err != nil {
		return nil, err
	}

	if _, err := InitializeSchema(New(db)); err != nil {
		return nil, err
	}

	return New(db), nil
}

func InitializeSchema(db *Queries) (sql.Result, error) {
	return db.db.ExecContext(context.Background(), schema)
}
