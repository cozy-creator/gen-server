package tools

import (
	"errors"
	"os/exec"
)

func CommandExists(command string) (bool, error) {
	if _, err := exec.LookPath(command); err != nil {
		return false, err
	}
	return true, nil
}

func GetPythonSitePackagesPath() (string, error) {
	pythonBins := []string{"python", "python3"}
	codeString := "import sysconfig; print(sysconfig.get_paths()['purelib'])"

	for _, pythonBin := range pythonBins {
		if _, err := CommandExists(pythonBin); err == nil {
			cmd := exec.Command(pythonBin, "-c", codeString)
			output, err := cmd.Output()
			if err != nil {
				return "", err
			}

			if len(output) == 0 {
				return "", errors.New("could not get python site packages path")
			}

			return string(output), nil
		}
	}

	// If none of the python bins exist, return an error
	return "", errors.New("python may not be installed, please check and try again")
}
