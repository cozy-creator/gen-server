package repository

import (
	"context"

	"github.com/cozy-creator/gen-server/internal/db/models"
	"github.com/uptrace/bun"
)

func GetModels(ctx context.Context, db *bun.DB, modelNames []string) ([]models.Model, error) {
	var models []models.Model
	err := db.NewSelect().
		Model(&models).
		Where("name IN (?)", bun.In(modelNames)).
		Scan(ctx)
	return models, err
}