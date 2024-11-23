package drivers

import (
	"context"
	"database/sql"
	"os"
	"path/filepath"
	"strings"

	"github.com/uptrace/bun"
	"github.com/uptrace/bun/dialect/sqlitedialect"

	_ "github.com/tursodatabase/libsql-client-go/libsql"
)

type SQLiteDriver struct {
	db *bun.DB
}

func NewSQLiteDriver(ctx context.Context, name, dsn string) (*SQLiteDriver, error) {
	if strings.HasPrefix(dsn, "file:") {
		dbpath, err := filepath.Abs(strings.TrimPrefix(dsn, "file:"))
		if err != nil {
			return nil, err
		}

		if _, err := os.Stat(dbpath); err != nil {
			if os.IsNotExist(err) {
				if err := os.MkdirAll(filepath.Dir(dbpath), os.ModePerm); err != nil {
					return nil, err
				}
				file, err := os.Create(dbpath)
				if err != nil {
					return nil, err
				}

				file.Close()
			} else {
				return nil, err
			}
		}
	}

	sqldb, err := sql.Open(name, dsn)
	if err != nil {
		return nil, err
	}

	return &SQLiteDriver{db: bun.NewDB(sqldb, sqlitedialect.New())}, nil
}

func (d *SQLiteDriver) GetDB() *bun.DB {
	return d.db
}
