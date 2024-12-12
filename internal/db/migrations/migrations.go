package migrations

import (
	"fmt"

	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/uptrace/bun/migrate"
)

var Migrations = migrate.NewMigrations()

func InitMigrations() error {
	cfg := config.MustGetConfig()

	if cfg != nil && cfg.Environment != "prod" {
		if err := Migrations.DiscoverCaller(); err != nil {
			fmt.Println("Error discovering caller: ", err)
			return err
		}
	}

	return nil
}
