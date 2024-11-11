package repository

import (
	"context"

	"github.com/cozy-creator/gen-server/internal/db/models"
	"github.com/uptrace/bun"
)

type IImageRepository interface {
	Repository[models.Image]
}

type ImageRepository struct {
	db *bun.DB
}

func NewImageRepository(db *bun.DB) IImageRepository {
	return &ImageRepository{db: db}
}

func (r *ImageRepository) Create(ctx context.Context, image *models.Image) (*models.Image, error) {
	if err := r.db.NewInsert().Model(image).Returning("*").Scan(ctx); err != nil {
		return nil, err
	}

	return image, nil
}

func (r *ImageRepository) GetByID(ctx context.Context, id string) (*models.Image, error) {
	var image models.Image
	if err := r.db.NewSelect().Model(&image).Where("id = ?", id).Scan(ctx); err != nil {
		return nil, err
	}

	return &image, nil
}

func (r *ImageRepository) UpdateByID(ctx context.Context, id string, image *models.Image) (*models.Image, error) {
	if err := r.db.NewUpdate().Model(image).Where("id = ?", id).Returning("*").Scan(ctx); err != nil {
		return nil, err
	}

	return image, nil
}

func (r *ImageRepository) DeleteByID(ctx context.Context, id string) error {
	_, err := r.db.NewDelete().Model(&models.Image{}).Where("id = ?", id).Exec(ctx)
	return err
}
