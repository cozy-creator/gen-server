package main

import (
	"cozy-creator/go-cozy/cmd"
	"fmt"
)

func main() {
	cli := cmd.NewCLI()

	if err := cli.Run(); err != nil {
		fmt.Println(err)
		return
	}
}
