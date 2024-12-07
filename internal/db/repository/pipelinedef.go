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

// type IPipelineDefRepository interface {
// 	Repository[models.PipelineDef]
// 	GetPipelineDefs(ctx context.Context, pipelineNames []string) ([]models.PipelineDef, error)
// }

// type PipelineDefRepository struct {
// 	db bun.IDB
// }

// func NewPipelineDefRepository(db *bun.DB) IPipelineDefRepository {
// 	return &PipelineDefRepository{db: db}
// }

// func (r *PipelineDefRepository) GetPipelineDefs(ctx context.Context, pipelineNames []string) ([]models.PipelineDef, error) {
// 	var pipelineDefs []models.PipelineDef

// 	err := r.db.NewSelect().
// 		Model(&pipelineDefs).
// 		Where("name IN (?)", bun.In(pipelineNames)).
// 		Scan(ctx)

// 	return pipelineDefs, err
// }

// func (r *PipelineDefRepository) Create(ctx context.Context, pipelineDef *models.PipelineDef) (*models.PipelineDef, error) {
// 	_, err := r.db.NewInsert().
// 		Model(&pipelineDef).
// 		Exec(ctx)
// 	return pipelineDef, err
// }

// func (r *PipelineDefRepository) GetByID(ctx context.Context, id string) (*models.PipelineDef, error) {
// 	var pipelineDef models.PipelineDef
// 	err := r.db.NewSelect().Model(&pipelineDef).Where("id = ?", id).Scan(ctx)
// 	return &pipelineDef, err
// }

// func (r *PipelineDefRepository) UpdateByID(ctx context.Context, id string, pipelineDef *models.PipelineDef) (*models.PipelineDef, error) {
// 	_, err := r.db.NewUpdate().Model(pipelineDef).Where("id = ?", id).Exec(ctx)
// 	return pipelineDef, err
// }

// func (r *PipelineDefRepository) DeleteByID(ctx context.Context, id string) error {
// 	_, err := r.db.NewDelete().Model(&models.PipelineDef{}).Where("id = ?", id).Exec(ctx)
// 	return err
// }

// func (r *PipelineDefRepository) List(ctx context.Context) ([]models.PipelineDef, error) {
// 	var pipelineDefs []models.PipelineDef
// 	err := r.db.NewSelect().Model(&pipelineDefs).Scan(ctx)
// 	return pipelineDefs, err
// }

