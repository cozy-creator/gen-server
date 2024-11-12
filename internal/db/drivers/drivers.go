package drivers

import "github.com/uptrace/bun"

type Driver interface {
	GetDB() *bun.DB
}
