package main

import (
	"fmt"
	"os"

	cmd "github.com/cozy-creator/gen-server/cmd/cozy"
)

func main() {
	rootCmd := cmd.GetRootCmd()
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
