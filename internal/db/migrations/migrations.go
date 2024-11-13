package migrations

import (
	"fmt"

	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/uptrace/bun/migrate"
)

var Migrations = migrate.NewMigrations()

func InitMigrations() error {
	cfg := config.GetConfig()

	if cfg != nil && cfg.Environment != "production" {
		if err := Migrations.DiscoverCaller(); err != nil {
			fmt.Println("Error discovering caller: ", err)
			return err
		}
	}

	return nil
}
