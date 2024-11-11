package db

import (
	"context"
	"fmt"

	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/db/drivers"
)

func NewConnection(ctx context.Context, config *config.Config) (drivers.Driver, error) {
	driver := config.DB.Driver

	if driver == "sqlite" {
		return drivers.NewSQLiteDriver(ctx, config.DB.DSN)
	} else if driver == "pg" {
		return drivers.NewPGDriver(ctx, config.DB.DSN)
	}

	return nil, fmt.Errorf("invalid database driver: %s", driver)
}
