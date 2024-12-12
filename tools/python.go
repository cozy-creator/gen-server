package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/config"
)

const RUNTIME_COMMAND = "cozy-runtime"

func CommandExists(command string) (bool, error) {
	if _, err := exec.LookPath(command); err != nil {
		return false, err
	}
	return true, nil
}

func CreateCozyRuntimeCommand(args ...string) (*exec.Cmd, error) {
	if _, err := CommandExists(RUNTIME_COMMAND); err == nil {
		cmd := exec.Command(RUNTIME_COMMAND, args...)
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		cmd.Env = os.Environ()

		return cmd, nil
	}

	return nil, fmt.Errorf("the command 'cozy-runtime' is not installed in the system-path; please install it")
}

func StartPythonRuntime(ctx context.Context, app *app.App, cfg *config.Config) error {
	ctx = context.WithoutCancel(ctx)

	pipelineDefsJson, err := json.Marshal(cfg.PipelineDefs)
    if err != nil {
        log.Printf("failed to marshal pipeline defs: %v", err)
        pipelineDefsJson = []byte{}
    }

	args := []string{
		"--home-dir", cfg.CozyHome,
		"--environment", cfg.Environment,
		"--host", "localhost", // we are not using cfg.Host, because cozy-runtime is accessible internally only
		"--port", strconv.Itoa(config.TCPPort),
		"--models-path", cfg.ModelsDir,
	}

	if len(cfg.EnabledModels) > 0 {
		args = append(args, "--enabled-models", strings.Join(cfg.EnabledModels, ","))
	}
	if len(pipelineDefsJson) > 0 {
		args = append(args, "--pipeline-defs", string(pipelineDefsJson))
	}

	cmd, err := CreateCozyRuntimeCommand(args...)
	if err != nil {
		return err
	}

	// Start the command
	if err := cmd.Start(); err != nil {
		return err
	}

	go func() {
		for {
			select {
			case <-ctx.Done():
				fmt.Println("Stopping Python Runtime...")
				cmd.Process.Kill()
				return
			default:
				time.Sleep(time.Second)
			}
		}
	}()

	// Wait for the command to finish
	if err := cmd.Wait(); err != nil {
		if ctx.Err() != nil && errors.Is(ctx.Err(), context.Canceled) {
			fmt.Println("Python Runtime stopped successfully")
			return nil
		}

		fmt.Println("Python Runtime stopped unexpectedly")

		return fmt.Errorf("error waiting for Python Runtime to exit: %w", err)
	}

	return nil
}


// ==== The rest of the code in this file is mostly useless ====

func CreatePythonCommand(args ...string) (*exec.Cmd, error) {
	// First check if we're in a venv
	venvPath := os.Getenv("VIRTUAL_ENV")
	if venvPath != "" {
		// Use the Python from the active venv
		var pythonPath string
		if runtime.GOOS == "windows" {
			pythonPath = filepath.Join(venvPath, "Scripts", "python.exe")
		} else {
			pythonPath = filepath.Join(venvPath, "bin", "python")
		}

		// Verify the file exists
		if _, err := os.Stat(pythonPath); err == nil {
			cmd := exec.Command(pythonPath, args...)
			cmd.Stdin = os.Stdin
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr

			// Ensure venv environment is properly set
			cmd.Env = append(os.Environ(),
				fmt.Sprintf("VIRTUAL_ENV=%s", venvPath),
				fmt.Sprintf("PATH=%s%c%s", 
					filepath.Join(venvPath, "bin"), 
					os.PathListSeparator, 
					os.Getenv("PATH")),
			)

			return cmd, nil
		} else {
			fmt.Printf("python.exe not found at %s\n", pythonPath)
		}
	}

	fmt.Println("No venv found; using global system Python")

	// Fallback to system Python if no venv is active
	commands := []string{"python", "python3"}
	for _, pythonBin := range commands {
		if _, err := CommandExists(pythonBin); err == nil {
			cmd := exec.Command(pythonBin, args...)
			cmd.Stdin = os.Stdin
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			cmd.Env = os.Environ()

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