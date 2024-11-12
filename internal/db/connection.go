package db

import (
	"context"
	"fmt"
	"strings"

	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/db/drivers"
	"github.com/uptrace/bun/driver/sqliteshim"
)

func NewConnection(ctx context.Context, config *config.Config) (drivers.Driver, error) {
	driver, err := detectDriver(config.DB.DSN)
	if err != nil {
		return nil, err
	}

	dsn := config.DB.DSN
	if driver == "sqlite" {
		return drivers.NewSQLiteDriver(ctx, sqliteshim.ShimName, dsn)
	}

	if driver == "libsql" {
		return drivers.NewSQLiteDriver(ctx, "libsql", dsn)
	}

	if driver == "postgres" {
		return drivers.NewPGDriver(ctx, dsn)
	}

	return nil, fmt.Errorf("invalid database driver: %s", driver)
}

func detectDriver(dsn string) (string, error) {
	if strings.HasPrefix(dsn, "libsql://") {
		return "libsql", nil
	}

	if strings.HasPrefix(dsn, "mysql://") || strings.Contains(dsn, "@tcp(") {
		return "mysql", nil
	}

	if strings.HasPrefix(dsn, "postgres://") || strings.HasPrefix(dsn, "postgresql://") {
		return "postgres", nil
	}

	if strings.HasPrefix(dsn, "sqlite3://") || strings.HasPrefix(dsn, "sqlite://") || strings.HasSuffix(dsn, ".db") || strings.Contains(dsn, ":memory:") {
		return "sqlite", nil
	}

	return "", fmt.Errorf("unknown database driver")
}
