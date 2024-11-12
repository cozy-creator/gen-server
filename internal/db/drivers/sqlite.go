package drivers

import (
	"context"
	"database/sql"

	"github.com/uptrace/bun"
	"github.com/uptrace/bun/dialect/sqlitedialect"

	_ "github.com/tursodatabase/libsql-client-go/libsql"
)

type SQLiteDriver struct {
	db *bun.DB
}

func NewSQLiteDriver(ctx context.Context, name, dsn string) (*SQLiteDriver, error) {
	sqldb, err := sql.Open(name, dsn)
	if err != nil {
		return nil, err
	}

	return &SQLiteDriver{db: bun.NewDB(sqldb, sqlitedialect.New())}, nil
}

func (d *SQLiteDriver) GetDB() *bun.DB {
	return d.db
}
