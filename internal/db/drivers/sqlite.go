package drivers

import (
	"context"
	"database/sql"

	"github.com/uptrace/bun"
	"github.com/uptrace/bun/dialect/sqlitedialect"
	"github.com/uptrace/bun/driver/sqliteshim"
)

type SQLiteDriver struct {
	db *bun.DB
}

func NewSQLiteDriver(ctx context.Context, dsn string) (*SQLiteDriver, error) {
	sqldb, err := sql.Open(sqliteshim.ShimName, dsn)
	if err != nil {
		return nil, err
	}

	return &SQLiteDriver{db: bun.NewDB(sqldb, sqlitedialect.New())}, nil
}

func (d *SQLiteDriver) GetDB() *bun.DB {
	return d.db
}
