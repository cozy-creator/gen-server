package repository

import (
	"context"
	"fmt"

	"github.com/cozy-creator/gen-server/internal/db/models"
	"github.com/uptrace/bun"
)

type IAPIKeyRepository interface {
	Repository[models.APIKey]
	WithTx(tx *bun.Tx) IAPIKeyRepository
	WithDB(db *bun.DB) IAPIKeyRepository
	RevokeAPIKeyWithHash(ctx context.Context, keyHash string) error
	GetAPIKeyWithHash(ctx context.Context, keyHash string) (*models.APIKey, error)
	ListAPIKeys(ctx context.Context) ([]models.APIKey, error)
}

type APIKeyRepository struct {
	db bun.IDB
}

func NewAPIKeyRepository(db *bun.DB) IAPIKeyRepository {
	return &APIKeyRepository{db: db}
}

func (r *APIKeyRepository) Create(ctx context.Context, apikey *models.APIKey) (*models.APIKey, error) {
	if apikey == nil {
		return nil, fmt.Errorf("apikey model is nil")
	}

	if err := r.db.NewInsert().Model(apikey).Returning("*").Scan(ctx); err != nil {
		return nil, err
	}

	return apikey, nil
}

func (r *APIKeyRepository) GetByID(ctx context.Context, id string) (*models.APIKey, error) {
	var apikey models.APIKey
	if err := r.db.NewSelect().Model(&apikey).Where("id = ?", id).Scan(ctx); err != nil {
		return nil, err
	}

	return &apikey, nil
}

func (r *APIKeyRepository) UpdateByID(ctx context.Context, id string, apikey *models.APIKey) (*models.APIKey, error) {
	if apikey == nil {
		return nil, fmt.Errorf("apikey model is nil")
	}

	if err := r.db.NewUpdate().Model(apikey).Where("id = ?", id).Returning("*").Scan(ctx); err != nil {
		return nil, err
	}

	return apikey, nil
}

func (r *APIKeyRepository) DeleteByID(ctx context.Context, id string) error {
	_, err := r.db.NewDelete().Model(&models.APIKey{}).Where("id = ?", id).Exec(ctx)
	return err
}

func (r *APIKeyRepository) RevokeAPIKeyWithHash(ctx context.Context, keyHash string) error {
	_, err := r.db.NewUpdate().Model(&models.APIKey{}).Where("key_hash = ?", keyHash).Set("is_revoked = ?", true).Exec(ctx)
	return err
}

func (r *APIKeyRepository) GetAPIKeyWithHash(ctx context.Context, keyHash string) (*models.APIKey, error) {
	var apiKey models.APIKey
	if err := r.db.NewSelect().Model(&apiKey).Where("key_hash = ?", keyHash).Scan(ctx); err != nil {
		return nil, err
	}

	return &apiKey, nil
}

func (r *APIKeyRepository) ListAPIKeys(ctx context.Context) ([]models.APIKey, error) {
	var apiKeys []models.APIKey
	if err := r.db.NewSelect().Model(&apiKeys).Scan(ctx); err != nil {
		return nil, err
	}

	return apiKeys, nil
}

func (r *APIKeyRepository) WithTx(tx *bun.Tx) IAPIKeyRepository {
	return &APIKeyRepository{db: tx}
}

func (r *APIKeyRepository) WithDB(db *bun.DB) IAPIKeyRepository {
	return &APIKeyRepository{db: db}
}
