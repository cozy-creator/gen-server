package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/gen-server/internal/config"
)

func CommandExists(command string) (bool, error) {
	if _, err := exec.LookPath(command); err != nil {
		return false, err
	}
	return true, nil
}

func ExecutePythonCommand(args ...string) (*exec.Cmd, error) {
	commands := []string{"python", "python3"}

	for _, pythonBin := range commands {
		if _, err := CommandExists(pythonBin); err == nil {
			cmd := exec.Command(pythonBin, args...)

			cmd.Stdin = os.Stdin
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr

			cmd.Env = os.Environ()

			err := cmd.Run()
			if err != nil {
				return nil, err
			}

			return cmd, nil
		}
	}

	return nil, fmt.Errorf("python may not be installed, please check and try again")
}

func ExecutePythonCommandWithOutput(args ...string) (string, error) {
	commands := []string{"python", "python3"}

	for _, pythonBin := range commands {
		if _, err := CommandExists(pythonBin); err == nil {
			cmd := exec.Command(pythonBin, args...)
			output, err := cmd.Output()
			if err != nil {
				return "", err
			}

			if len(output) == 0 {
				return "", errors.New("could not get python site packages path")
			}

			return strings.TrimSpace(string(output)), nil
		}
	}

	return "", fmt.Errorf("python may not be installed, please check and try again")
}

func GetPythonSitePackagesPath() (string, error) {
	codeString := "import sysconfig; print(sysconfig.get_paths()['purelib'])"
	return ExecutePythonCommandWithOutput("-c", codeString)
}

func ResolveGenServerPath(version string) (string, error) {
	sitePackages, err := GetPythonSitePackagesPath()
	if err != nil {
		return "", err
	}

	// If gen-server is in site-packages, it's probably installed in non-editable mode
	genServerPath := filepath.Join(sitePackages, "gen_server")
	if _, err := os.Stat(genServerPath); err == nil {
		return genServerPath, nil
	}

	// If gen-server is not in site-packages, it's probably installed in editable mode,
	// so try to find it in the gen_server-<version>.dist-info directory
	genServerPath = filepath.Clean(filepath.Join(sitePackages, fmt.Sprintf("gen_server-%s.dist-info", version)))
	if _, err := os.Stat(genServerPath); err == nil {
		content, err := os.ReadFile(filepath.Join(genServerPath, "direct_url.json"))
		if err != nil {
			return "", err
		}

		var directUrl map[string]any
		if err := json.Unmarshal(content, &directUrl); err != nil {
			return "", err
		}
		if directUrl["url"] == nil {
			return "", fmt.Errorf("direct_url.json does not contain a url key")
		}

		urlPath, err := StripFilePrefix(directUrl["url"].(string))
		if err != nil {
			return "", err
		}

		genServerPath = filepath.Join(urlPath, "src", "gen_server")
		if _, err := os.Stat(genServerPath); err != nil {
			return "", err
		}

		return filepath.Clean(genServerPath), nil
	}

	return "", fmt.Errorf("gen-server not found in site-packages")
}

func StartPythonGenServer(ctx context.Context, version string, cfg *config.Config) error {
	ctx = context.WithoutCancel(ctx)
	genServerPath, err := ResolveGenServerPath(version)
	if err != nil {
		return err
	}

	mainFilePath := filepath.Join(genServerPath, "main.py")
	if _, err := os.Stat(mainFilePath); err != nil {
		return fmt.Errorf("main.py not found in gen-server path")
	}

	fmt.Println("Starting Python Gen Server. Models to start with:", cfg.WarmupModels)
	cmd, err := ExecutePythonCommand(
		"-m",
		mainFilePath,
		"--environment", cfg.Environment,
		"--host", cfg.Host,
		"--port", strconv.Itoa(cfg.TcpPort),
        "--warmup-models", strings.Join(cfg.WarmupModels, ","),
		"--models-path", cfg.ModelsDir,
	)
	if err != nil {
		return err
	}

	go func() {
		for {
			select {
			case <-ctx.Done():
				fmt.Println("Stopping Python Gen Server...")
				cmd.Process.Kill()
				return
			default:
				time.Sleep(time.Second)
			}
		}
	}()

	if err := cmd.Wait(); err != nil {
		if ctx.Err() != nil && errors.Is(ctx.Err(), context.Canceled) {
			fmt.Println("Python Gen Server stopped successfully")
			return nil
		}

		fmt.Println("Python Gen Server stopped unexpectedly")

		return fmt.Errorf("error waiting for Python Gen Server to exit: %w", err)
	}

	return nil
}

func StripFilePrefix(uri string) (string, error) {
	const fileScheme = "file://"

	if !strings.HasPrefix(uri, fileScheme) {
		return "", fmt.Errorf("invalid file URI: %s", uri)
	}

	uri = strings.TrimPrefix(uri, fileScheme)

	// Handle Windows paths
	if runtime.GOOS == "windows" {
		// Replace forward slashes with backslashes
		uri = strings.ReplaceAll(uri, "/", "\\")

		// Check if the path starts with a drive letter (e.g., C:)
		if len(uri) > 2 && uri[1] == ':' {
			return uri, nil
		}

		// Handle UNC paths (file://server/share/path)
		if strings.HasPrefix(uri, "\\\\") {
			return uri, nil
		}

		// Remove any leading backslashes that were added unnecessarily
		uri = strings.TrimPrefix(uri, "\\")

		// Prepend the UNC path prefix for absolute paths
		if strings.HasPrefix(uri, "\\") {
			return "\\" + uri, nil
		}
	} else {
		// On Unix-like systems, decode any URL-encoded characters
		decoded, err := url.PathUnescape(uri)
		if err != nil {
			return "", fmt.Errorf("error unescaping path: %w", err)
		}
		return decoded, nil
	}

	return uri, nil
}
