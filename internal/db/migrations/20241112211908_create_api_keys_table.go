package migrations

import (
	"context"

	"github.com/cozy-creator/gen-server/internal/db/models"
	"github.com/uptrace/bun"
)

func init() {
	Migrations.MustRegister(func(ctx context.Context, db *bun.DB) error {
		if _, err := db.NewCreateTable().Model((*models.APIKey)(nil)).IfNotExists().Exec(ctx); err != nil {
			return err
		}
		return nil
	}, func(ctx context.Context, db *bun.DB) error {
		if _, err := db.NewDropTable().Model((*models.APIKey)(nil)).IfExists().Exec(ctx); err != nil {
			return err
		}
		return nil
	})
}
