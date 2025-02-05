package cmd

import (
	"context"
	"fmt"

	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/db"
	"github.com/cozy-creator/gen-server/internal/db/migrations"
	"github.com/uptrace/bun/extra/bundebug"
	"github.com/uptrace/bun/migrate"

	"github.com/spf13/cobra"
)

type dbContextKey string
const migratorKey dbContextKey = "migrator"

var Cmd = &cobra.Command{
	Use:   "db",
	Short: "Utility for database management",
	PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
		driver, err := db.NewConnection(cmd.Context(), config.MustGetConfig())
		if err != nil {
			return err
		}

		db := driver.GetDB()
		db.AddQueryHook(bundebug.NewQueryHook(
			bundebug.WithEnabled(false),
			bundebug.FromEnv(),
		))

		if err := migrations.InitMigrations(); err != nil {
			return err
		}

		migrator := migrate.NewMigrator(db, migrations.Migrations)
		cmd.SetContext(context.WithValue(cmd.Context(), migratorKey, migrator))
		return nil
	},
}

func init() {
	setupMigrationCmd(Cmd)
}

func setupMigrationCmd(cmd *cobra.Command) error {
	migrationCmd := &cobra.Command{
		Use:   "migration",
		Short: "Utility for handling database migrations",
	}

	initCmd := &cobra.Command{
		Use:   "init",
		Short: "create migration tables",
		RunE: func(cmd *cobra.Command, args []string) error {
			migrator := cmd.Context().Value(migratorKey).(*migrate.Migrator)
			return migrator.Init(cmd.Context())
		},
	}

	migrateCmd := &cobra.Command{
		Use:   "migrate",
		Short: "migrate database",
		RunE: func(cmd *cobra.Command, args []string) error {
			migrator := cmd.Context().Value(migratorKey).(*migrate.Migrator)

			if err := migrator.Lock(cmd.Context()); err != nil {
				return err
			}
			defer migrator.Unlock(cmd.Context()) //nolint:errcheck

			group, err := migrator.Migrate(cmd.Context())
			if err != nil {
				return err
			}
			if group.IsZero() {
				fmt.Printf("there are no new migrations to run (database is up to date)\n")
				return nil
			}
			fmt.Printf("migrated to %s\n", group)
			return nil
		},
	}

	rollbackCmd := &cobra.Command{
		Use:   "rollback",
		Short: "rollback the last migration group",
		RunE: func(cmd *cobra.Command, args []string) error {
			migrator := cmd.Context().Value(migratorKey).(*migrate.Migrator)

			if err := migrator.Lock(cmd.Context()); err != nil {
				return err
			}
			defer migrator.Unlock(cmd.Context()) //nolint:errcheck

			group, err := migrator.Rollback(cmd.Context())
			if err != nil {
				return err
			}
			if group.IsZero() {
				fmt.Printf("there are no groups to roll back\n")
				return nil
			}
			fmt.Printf("rolled back %s\n", group)
			return nil
		},
	}

	lockCmd := &cobra.Command{
		Use:   "lock",
		Short: "Lock the database",
		RunE: func(cmd *cobra.Command, args []string) error {
			migrator := cmd.Context().Value(migratorKey).(*migrate.Migrator)

			if err := migrator.Lock(cmd.Context()); err != nil {
				return err
			}
			fmt.Printf("locked\n")
			return nil
		},
	}

	unlockCmd := &cobra.Command{
		Use:   "unlock",
		Short: "Unlock the database",
		RunE: func(cmd *cobra.Command, args []string) error {
			migrator := cmd.Context().Value(migratorKey).(*migrate.Migrator)

			if err := migrator.Unlock(cmd.Context()); err != nil {
				return err
			}
			fmt.Printf("unlocked\n")
			return nil
		},
	}

	createGoCmd := &cobra.Command{
		Use:   "create-go",
		Short: "Create a Go migration file",
		RunE: func(cmd *cobra.Command, args []string) error {
			migrator := cmd.Context().Value(migratorKey).(*migrate.Migrator)

			file, err := migrator.CreateGoMigration(cmd.Context(), args[0])
			if err != nil {
				fmt.Println(err)
				return err
			}

			fmt.Printf("created migration file %s in %s\n", file.Name, file.Path)
			return nil
		},
	}

	statusCmd := &cobra.Command{
		Use:   "status",
		Short: "Print the status of the migrations",
		RunE: func(cmd *cobra.Command, args []string) error {
			migrator := cmd.Context().Value(migratorKey).(*migrate.Migrator)

			status, err := migrator.MigrationsWithStatus(cmd.Context())
			if err != nil {
				return err
			}
			fmt.Printf("migrations: %s\n", status)
			return nil
		},
	}

	markAppliedCmd := &cobra.Command{
		Use:   "mark-applied",
		Short: "Mark all migrations as applied without actually running them",
		RunE: func(cmd *cobra.Command, args []string) error {
			migrator := cmd.Context().Value(migratorKey).(*migrate.Migrator)

			group, err := migrator.Migrate(cmd.Context(), migrate.WithNopMigration())
			if err != nil {
				return err
			}
			if group.IsZero() {
				fmt.Printf("there are no new migrations to mark as applied\n")
				return nil
			}
			fmt.Printf("marked as applied %s\n", group)
			return nil
		},
	}

	migrationCmd.AddCommand(
		initCmd,
		migrateCmd,
		rollbackCmd,
		lockCmd,
		unlockCmd,
		createGoCmd,
		statusCmd,
		markAppliedCmd,
	)

	cmd.AddCommand(migrationCmd)

	return nil
}
