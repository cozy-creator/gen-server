package migrations

import (
	"context"

	"github.com/cozy-creator/gen-server/internal/db/models"
	"github.com/uptrace/bun"
)

func init() {
	Migrations.MustRegister(func(ctx context.Context, db *bun.DB) error {
		if _, err := db.NewCreateTable().
			Model((*models.PromptDef)(nil)).
			IfNotExists().
			Exec(ctx); err != nil {
			return err
		}

		if _, err := db.Exec(`
			ALTER TABLE pipeline_defs
			ADD COLUMN IF NOT EXISTS prompt_def_id UUID REFERENCES prompt_defs(id);
		`); err != nil {
			return err
		}

		return nil
	}, func(ctx context.Context, db *bun.DB) error {
		if _, err := db.Exec(`
			ALTER TABLE pipeline_defs
			DROP COLUMN IF EXISTS prompt_def_id;
		`); err != nil {
			return err
		}

		if _, err := db.NewDropTable().
			Model((*models.PromptDef)(nil)).
			IfExists().
			Exec(ctx); err != nil {
			return err
		}

		return nil
	})
}
