# imgctl Client

`imgctl` is a command-line interface (CLI) for interacting with the Image Service Backend. It allows you to view container images from both the local containerd runtime and a remote Harbor registry. `imgctl` client will be installed on the head node of the BCM cluster.

## Dependencies

-   `curl`: For making HTTP requests to the server.
-   `jq`: For parsing and formatting JSON data. The install script will attempt to install this if it's missing.

## Installation
Give execution permission to install.sh
```bash
chmod +x install.sh
```

An installation script, `install.sh`, is provided to simplify setup.

1.  **Navigate to the client directory:**
    ```bash
    cd client
    ```

2.  **Run the installation script:**
    The script must be run with `sudo` because it copies the `imgctl` executable to `/usr/local/bin` and attempts to install `jq` using the system's package manager.

    ```bash
    sudo ./install.sh
    ```

## Configuration

The `imgctl` client needs to know the URL of the Image Service Backend.

-   **To set the server URL**, use the `config` command:
    ```bash
    imgctl config --server http://10.141.0.2:5375
    ```

-   **How it works:**
    -   The command saves the configuration to a user-specific file at `~/.config/imgctl/config`.
    -   If the user-specific file is not found, `imgctl` will look for a system-wide configuration at `/etc/imgctl/config`.
    -   If no configuration file is found, it falls back to the default URL: `http://10.141.0.2:5375`.

## Usage

```
Usage: imgctl COMMAND [OPTIONS]

Commands:
    get [local|harbor]     Get container images (default: all)
        Options:
            -o, --output FORMAT    Output format: json|table (default: table)

    config                 Manage configuration
        Options:
            --server URL           Set server URL

    help                   Show this help message

Examples:
    imgctl get                    # Get all images from local containerd and Harbor
    imgctl get local              # Get only local containerd images
    imgctl get harbor             # Get only Harbor registry images
    imgctl get -o json            # Get all images in JSON format
    imgctl config --server http://new.server:5000   # Set a new server URL
``` 
