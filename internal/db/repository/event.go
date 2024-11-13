package repository

import (
	"context"
	"fmt"

	"github.com/cozy-creator/gen-server/internal/db/models"
	"github.com/uptrace/bun"
)

type IEventRepository interface {
	Repository[models.Event]
	WithTx(tx *bun.Tx) IEventRepository
	WithDB(db *bun.DB) IEventRepository
}

type EventRepository struct {
	db bun.IDB
}

func NewEventRepository(db *bun.DB) IEventRepository {
	return &EventRepository{db: db}
}

func (r *EventRepository) Create(ctx context.Context, event *models.Event) (*models.Event, error) {
	if event == nil {
		return nil, fmt.Errorf("event model is nil")
	}

	if err := r.db.NewInsert().Model(event).Returning("*").Scan(ctx); err != nil {
		return nil, err
	}

	return event, nil
}

func (r *EventRepository) GetByID(ctx context.Context, id string) (*models.Event, error) {
	var event models.Event
	if err := r.db.NewSelect().Model(&event).Where("id = ?", id).Scan(ctx); err != nil {
		return nil, err
	}

	return &event, nil
}

func (r *EventRepository) UpdateByID(ctx context.Context, id string, event *models.Event) (*models.Event, error) {
	if event == nil {
		return nil, fmt.Errorf("event model is nil")
	}

	if err := r.db.NewUpdate().Model(event).Where("id = ?", id).Returning("*").Scan(ctx); err != nil {
		return nil, err
	}

	return event, nil
}

func (r *EventRepository) DeleteByID(ctx context.Context, id string) error {
	_, err := r.db.NewDelete().Model(&models.Event{}).Where("id = ?", id).Exec(ctx)
	return err
}

func (r *EventRepository) WithTx(tx *bun.Tx) IEventRepository {
	return &EventRepository{db: tx}
}

func (r *EventRepository) WithDB(db *bun.DB) IEventRepository {
	return &EventRepository{db: db}
}
