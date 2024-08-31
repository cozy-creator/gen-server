package main

import (
	cmd "cozy-creator/gen-server/cmd/cozy"
	"fmt"
	"os"
)

func main() {
	rootCmd := cmd.GetRootCmd()
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	// err := tools.StartPythonGenServer("0.2.2")
	// if err != nil {
	// 	fmt.Println(err)
	// 	os.Exit(1)
	// }
}
