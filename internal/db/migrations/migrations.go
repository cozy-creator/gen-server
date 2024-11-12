package migrations

import (
	"fmt"

	"github.com/uptrace/bun/migrate"
)

var Migrations = migrate.NewMigrations()

func init() {
	if err := Migrations.DiscoverCaller(); err != nil {
		fmt.Println("Error discovering caller: ", err)
	}
}
