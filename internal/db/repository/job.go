package repository

import (
	"context"
	"fmt"

	"github.com/cozy-creator/gen-server/internal/db/models"
	"github.com/uptrace/bun"
)

type IJobRepository interface {
	Repository[models.Job]
	UpdateJobStatusByID(ctx context.Context, id string, status models.JobStatus) error
	// GetWithOutputByID(ctx context.Context, id string) (*models.Job, error)
}

type JobRepository struct {
	db *bun.DB
}

func NewJobRepository(db *bun.DB) IJobRepository {
	return &JobRepository{db: db}
}

func (r *JobRepository) Create(ctx context.Context, job *models.Job) (*models.Job, error) {
	if job == nil {
		return nil, fmt.Errorf("job model is nil")
	}

	if err := r.db.NewInsert().Model(job).Returning("*").Scan(ctx); err != nil {
		return nil, err
	}

	return job, nil
}

func (r *JobRepository) GetByID(ctx context.Context, id string) (*models.Job, error) {
	var job models.Job
	if err := r.db.NewSelect().Model(&job).Where("id = ?", id).Scan(ctx); err != nil {
		return nil, err
	}

	return &job, nil
}

func (r *JobRepository) UpdateByID(ctx context.Context, id string, job *models.Job) (*models.Job, error) {
	if job == nil {
		return nil, fmt.Errorf("job model is nil")
	}

	if err := r.db.NewUpdate().Model(job).Where("id = ?", id).Returning("*").Scan(ctx); err != nil {
		return nil, err
	}

	return job, nil
}

func (r *JobRepository) DeleteByID(ctx context.Context, id string) error {
	_, err := r.db.NewDelete().Model(&models.Job{}).Where("id = ?", id).Exec(ctx)
	return err
}

func (r *JobRepository) UpdateJobStatusByID(ctx context.Context, id string, status models.JobStatus) error {
	_, err := r.db.NewUpdate().Model(&models.Job{}).Where("id = ?", id).Set("status = ?", status).Exec(ctx)
	return err
}
