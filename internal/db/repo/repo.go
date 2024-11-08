package repo

import "context"

type Repo interface {
	Get(ctx context.Context, arg interface{}) error
	Create(ctx context.Context, arg interface{}) error
	Update(ctx context.Context, arg interface{}) error
	Delete(ctx context.Context, arg interface{}) error
}
