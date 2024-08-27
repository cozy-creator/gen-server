package tools

import (
	"errors"
	"os/exec"
)

func GetPythonSitePackagesPath() (string, error) {
	cmd := exec.Command("python3", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])")
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}

	if len(output) == 0 {
		return "", errors.New("could not get python site packages path")
	}

	return string(output), nil
}
