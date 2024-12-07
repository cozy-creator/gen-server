package repository

import (
	"context"

	"github.com/cozy-creator/gen-server/internal/db/models"
	"github.com/uptrace/bun"
)

func GetPipelineDefs(ctx context.Context, db *bun.DB, pipelineNames []string) ([]models.PipelineDef, error) {
	var pipelineDefs []models.PipelineDef

	err := db.NewSelect().
		Model(&pipelineDefs).
		Where("name IN (?)", bun.In(pipelineNames)).
		Scan(ctx)

	return pipelineDefs, err
}