package cmd

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/spf13/cobra"
)

const (
	defaultWebDir         = "web"
	defaultPackageManager = "npm"
)

var (
	webDir         string
	distDir        string
	packageManager string
)

var buildWebCmd = &cobra.Command{
	Use:   "build-web",
	Short: "Build the web bundle",
	RunE: func(cmd *cobra.Command, args []string) error {
		log.Println("Building web bundle...")

		if packageManager != "npm" && packageManager != "yarn" {
			return fmt.Errorf("package manager must be either 'npm' or 'yarn'")
		}

		if webDir == "" {
			webDir = defaultWebDir
		}

		if distDir == "" {
			distDir = filepath.Join(webDir, "dist")
		}

		// Verify the webDir exists
		if _, err := os.Stat(webDir); os.IsNotExist(err) {
			return fmt.Errorf("web directory does not exist: %s", webDir)
		}

		// Store the original working directory to return to it later
		originalDir, err := os.Getwd()
		if err != nil {
			return fmt.Errorf("could not get current directory: %w", err)
		}
		defer os.Chdir(originalDir) // Ensure we return to the original directory

		err = os.Chdir(webDir)
		if err != nil {
			log.Println("Error changing directory:", err)
			return fmt.Errorf("error changing directory to %s: %w", webDir, err)
		}

		{
			// Install the web dependencies
			log.Println("Installing web dependencies...")
			install := exec.Command(packageManager, "install")
			install.Stdout = os.Stdout
			install.Stderr = os.Stderr
			if err := install.Run(); err != nil {
				return fmt.Errorf("error running %s install: %w", packageManager, err)
			}
		}

		{
			// Build the web bundle
			log.Println("Building web bundle...")
			build := exec.Command(packageManager, "run", "build")
			build.Stdout = os.Stdout
			build.Stderr = os.Stderr
			if err := build.Run(); err != nil {
				return fmt.Errorf("error running %s build: %w", packageManager, err)
			}
		}
		log.Println("Web bundle built successfully.")
		return nil
	},
}

func init() {
	buildWebCmd.Flags().StringVar(&webDir, "web-dir", defaultWebDir, "The directory containing the web assets")
	buildWebCmd.Flags().StringVar(&distDir, "dist-dir", "", "The directory to store the built web bundle")
	buildWebCmd.Flags().StringVar(&packageManager, "package-manager", defaultPackageManager, "The package manager to use for building the web bundle")
	rootCmd.AddCommand(buildWebCmd)
}
