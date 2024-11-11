package drivers

import (
	"context"
	"database/sql"

	"github.com/uptrace/bun"
	"github.com/uptrace/bun/dialect/pgdialect"
	"github.com/uptrace/bun/driver/pgdriver"
)

type PGDriver struct {
	db *bun.DB
}

func NewPGDriver(ctx context.Context, dsn string) (*PGDriver, error) {
	sqldb := sql.OpenDB(pgdriver.NewConnector(pgdriver.WithDSN(dsn)))
	return &PGDriver{db: bun.NewDB(sqldb, pgdialect.New())}, nil
}

func (d *PGDriver) GetDB() *bun.DB {
	return d.db
}
