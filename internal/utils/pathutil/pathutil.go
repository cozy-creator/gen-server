package pathutil

import (
	"os"
	"path/filepath"
	"strings"
)

// ExpandPath expands the path using the user's home directory.
// If the path starts with "~", it is replaced with the user's home directory.
func ExpandPath(path string) (string, error) {
	if strings.HasPrefix(path, "~") {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}

		// Replace "~" with the home directory path
		path = filepath.Join(homeDir, path[1:])
	}

	return path, nil
}
