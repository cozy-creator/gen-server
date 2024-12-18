package cmd

import (
	"context"
	"fmt"

	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/db"
	"github.com/cozy-creator/gen-server/internal/db/models"
	"github.com/cozy-creator/gen-server/internal/db/repository"
	"github.com/cozy-creator/gen-server/internal/utils/hashutil"
	"github.com/cozy-creator/gen-server/internal/utils/randutil"
	"github.com/google/uuid"
	"github.com/spf13/cobra"
)

type contextKey string
const apiKeyRepoKey contextKey = "apikey_repo"

var Cmd = &cobra.Command{
	Use:   "api-key",
	Short: "Manage Cozy API keys",
	PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
		driver, err := db.NewConnection(cmd.Context(), config.MustGetConfig())
		if err != nil {
			return err
		}

		db := driver.GetDB()
		cmd.SetContext(context.WithValue(cmd.Context(), apiKeyRepoKey, repository.NewAPIKeyRepository(db)))
		return nil
	},
}

var newAPIKeyCmd = &cobra.Command{
	Use:   "new",
	Short: "Creates a new API key",
	RunE: func(cmd *cobra.Command, args []string) error {
		key, err := randutil.RandomString(32)
		if err != nil {
			return err
		}

		mask := randutil.MaskString(key, 4, 4)
		repo := cmd.Context().Value(apiKeyRepoKey).(*repository.APIKeyRepository)
		apiKey := models.APIKey{
			KeyMask:   mask,
			IsRevoked: false,
			ID:        uuid.Must(uuid.NewRandom()),
			KeyHash:   hashutil.Sha3256Hash([]byte(key)),
		}

		if _, err := repo.Create(cmd.Context(), &apiKey); err != nil {
			return err
		}

		fmt.Printf("API key created: %s\n", key)
		return nil
	},
}

var revokeAPIKeyCmd = &cobra.Command{
	Use:   "revoke",
	Short: "Revoke an API key",
	RunE: func(cmd *cobra.Command, args []string) error {
		key := args[0]
		repo := cmd.Context().Value(apiKeyRepoKey).(*repository.APIKeyRepository)

		if err := repo.RevokeAPIKeyWithHash(cmd.Context(), hashutil.Sha3256Hash([]byte(key))); err != nil {
			return err
		}

		fmt.Printf("API key revoked: %s\n", key)
		return nil
	},
}

var listAPIKeysCmd = &cobra.Command{
	Use:   "list",
	Short: "List all API keys",
	RunE: func(cmd *cobra.Command, args []string) error {
		repo := cmd.Context().Value(apiKeyRepoKey).(*repository.APIKeyRepository)

		apiKeys, err := repo.ListAPIKeys(cmd.Context())
		if err != nil {
			return err
		}

		if len(apiKeys) == 0 {
			fmt.Println("No API keys found")
			return nil
		}

		fmt.Println("API keys:")
		for _, apiKey := range apiKeys {
			fmt.Printf("%s (Revoked: %t)\n", apiKey.KeyMask, apiKey.IsRevoked)
		}

		return nil
	},
}

func init() {
	Cmd.AddCommand(newAPIKeyCmd, revokeAPIKeyCmd, listAPIKeysCmd)
}
