package repository

import "context"

type Repository[T any] interface {
	Create(ctx context.Context, arg *T) (*T, error)
	GetByID(ctx context.Context, id string) (*T, error)
	UpdateByID(ctx context.Context, id string, arg *T) (*T, error)
	DeleteByID(ctx context.Context, id string) error
}
