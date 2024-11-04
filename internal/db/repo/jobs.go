package repo

import (
	"context"
	"fmt"

	"github.com/cozy-creator/gen-server/internal/db"
	"github.com/google/uuid"
)

type JobsRepo struct {
	db *db.Queries
}

func NewJobsRepo(db *db.Queries) *JobsRepo {
	return &JobsRepo{db: db}
}

func (r *JobsRepo) Create(ctx context.Context, arg interface{}) (*db.Job, error) {
	data, ok := arg.(db.CreateJobParams)
	if !ok {
		return nil, fmt.Errorf("invalid argument type")
	}

	job, err := r.db.CreateJob(ctx, data)
	if err != nil {
		return nil, err
	}

	return &job, nil
}

func (r *JobsRepo) Get(ctx context.Context, arg interface{}) (*db.Job, error) {
	id, ok := arg.(uuid.UUID)
	if !ok {
		return nil, fmt.Errorf("invalid argument type")
	}

	job, err := r.db.GetJobByID(ctx, id)
	if err != nil {
		return nil, err
	}

	return &job, nil
}

func (r *JobsRepo) GetWithImages(ctx context.Context, arg interface{}) (*db.GetJobWithImagesRow, error) {
	id, ok := arg.(uuid.UUID)
	if !ok {
		return nil, fmt.Errorf("invalid argument type")
	}

	job, err := r.db.GetJobWithImages(ctx, id)
	if err != nil {
		return nil, err
	}

	return &job, nil
}

func (r *JobsRepo) UpdateStatus(ctx context.Context, arg interface{}) error {
	data, ok := arg.(db.UpdateJobStatusParams)
	if !ok {
		return fmt.Errorf("invalid argument type")
	}

	return r.db.UpdateJobStatus(ctx, data)
}
