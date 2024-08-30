package tools

import (
	"cozy-creator/gen-server/internal/config"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

func CommandExists(command string) (bool, error) {
	if _, err := exec.LookPath(command); err != nil {
		return false, err
	}
	return true, nil
}

func ExecutePythonCommand(args ...string) error {
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
				return err
			}

			return nil
		}
	}

	return fmt.Errorf("python may not be installed, please check and try again")
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

		genServerPath = filepath.Join(directUrl["url"].(string), "src", "gen_server")
		if strings.HasPrefix(genServerPath, "file:") {
			genServerPath = strings.TrimPrefix(genServerPath, "file:")
		}

		if _, err := os.Stat(genServerPath); err != nil {
			return "", err
		}

		return filepath.Clean(genServerPath), nil
	}

	return "", fmt.Errorf("gen-server not found in site-packages")
}

func StartPythonGenServer(version string, cfg *config.Config) error {
	genServerPath, err := ResolveGenServerPath(version)
	if err != nil {
		return err
	}

	mainFilePath := filepath.Join(genServerPath, "main.py")
	if _, err := os.Stat(mainFilePath); err != nil {
		return fmt.Errorf("main.py not found in gen-server path")
	}

	err = ExecutePythonCommand(mainFilePath, "--environment", cfg.Environment, "--port", strconv.Itoa(cfg.TcpPort))
	if err != nil {
		return err
	}

	return nil
}
