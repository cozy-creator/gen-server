package repo

import (
	"context"
	"fmt"

	"github.com/cozy-creator/gen-server/internal/db"
	"github.com/google/uuid"
)

type ImagesRepo struct {
	db *db.Queries
}

func NewImagesRepo(db *db.Queries) *ImagesRepo {
	return &ImagesRepo{db: db}
}

func (r *ImagesRepo) Create(ctx context.Context, arg interface{}) (*db.Image, error) {
	data, ok := arg.(db.CreateImageParams)
	if !ok {
		return nil, fmt.Errorf("invalid argument type")
	}

	job, err := r.db.CreateImage(ctx, data)
	if err != nil {
		return nil, err
	}

	return &job, nil
}

func (r *ImagesRepo) Get(ctx context.Context, arg interface{}) (*db.Image, error) {
	id, ok := arg.(uuid.UUID)
	if !ok {
		return nil, fmt.Errorf("invalid argument type")
	}

	job, err := r.db.GetImageByID(ctx, id)
	if err != nil {
		return nil, err
	}

	return &job, nil
}
