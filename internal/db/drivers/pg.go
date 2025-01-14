package drivers

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"github.com/uptrace/bun"
	"github.com/uptrace/bun/dialect/pgdialect"
	"github.com/uptrace/bun/driver/pgdriver"
)

type PGDriver struct {
	db *bun.DB
}

func NewPGDriver(ctx context.Context, dsn string) (*PGDriver, error) {
    connector := pgdriver.NewConnector(
        pgdriver.WithDSN(dsn),
        pgdriver.WithTimeout(20 * time.Second),        
        pgdriver.WithDialTimeout(5 * time.Second),     
        pgdriver.WithReadTimeout(10 * time.Second),    
        pgdriver.WithWriteTimeout(5 * time.Second),    
        pgdriver.WithApplicationName("cozy-gen"),
    )

    // Create the database connection
    sqldb := sql.OpenDB(connector)
    db := bun.NewDB(sqldb, pgdialect.New())

    // Configure connection pool on the bun.DB instance
    db.SetMaxOpenConns(25)
    db.SetMaxIdleConns(5)
    db.SetConnMaxLifetime(5 * time.Minute)
    db.SetConnMaxIdleTime(10 * time.Minute)

    // Test the connection
    if err := db.PingContext(ctx); err != nil {
        return nil, fmt.Errorf("failed to ping database: %w", err)
    }

    return &PGDriver{db: db}, nil
}

func (d *PGDriver) GetDB() *bun.DB {
	return d.db
}
