def onboard():
    """Print setup instructions for CI/CD, PyPI publishing, and GitHub configuration."""
    print(
        """
================================================================================
  surfvis — Setup Instructions
================================================================================

Follow these steps to complete the CI/CD and publishing setup for your project.

────────────────────────────────────────────────────────────────────────────────
  Step 1: Create a GitHub Repository
────────────────────────────────────────────────────────────────────────────────

  First, install the GitHub CLI (gh) if you haven't already:

    # macOS
    brew install gh

    # Ubuntu/Debian
    sudo apt install gh

    # Other platforms: https://github.com/cli/cli#installation

  Then authenticate:

    gh auth login

  Push your project to GitHub:

    gh repo create ratt-ru/surfvis --public --source=. --push

  Or create the repo manually at https://github.com/new and push:

    git remote add origin git@github.com:ratt-ru/surfvis.git
    git push -u origin master

────────────────────────────────────────────────────────────────────────────────
  Step 2: Set Up Trusted Publishing on PyPI
────────────────────────────────────────────────────────────────────────────────

  Use PyPI's "pending trusted publisher" feature to pre-authorize GitHub
  Actions to publish your package. The PyPI project will be created
  automatically on the first successful publish.

  Go to: https://pypi.org/manage/account/publishing/

  Scroll down to "Add a new pending publisher" and fill in:
    PyPI project name: surfvis
    Owner:             ratt-ru
    Repository:        surfvis
    Workflow:          publish.yml
    Environment:       pypi

────────────────────────────────────────────────────────────────────────────────
  Step 3: Create GitHub Environment
────────────────────────────────────────────────────────────────────────────────

  The publish workflow requires a GitHub environment named "pypi".

  Go to: https://github.com/ratt-ru/surfvis/settings/environments

  Click "New environment" and name it: pypi

────────────────────────────────────────────────────────────────────────────────
  Step 4: Create a GitHub App (for Automated Cab Updates)
────────────────────────────────────────────────────────────────────────────────

  The update-cabs workflow needs to push commits to the repository. A GitHub
  App is required so these commits can bypass branch protection rules.

  a) Create the app:

     Go to: https://github.com/settings/apps → "New GitHub App"

     Settings:
       Name:        surfvis-bot (or any name you like)
       Homepage:    https://github.com/ratt-ru/surfvis
       Webhook:     Uncheck "Active" (not needed)
       Permissions: Repository → Contents → Read & write
       Where:       Only on this account

  b) Generate a private key:

     On the app page → "Generate a private key"
     Save the downloaded .pem file.

  c) Install the app on your repository:

     On the app page → "Install App" → Select your repository.

  d) Add secrets to your repository:

     Go to: https://github.com/ratt-ru/surfvis/settings/secrets/actions

     Add two secrets:
       APP_CLIENT_ID   → The Client ID shown on the app's settings page
       APP_PRIVATE_KEY → The contents of the .pem file you downloaded

────────────────────────────────────────────────────────────────────────────────
  Step 5: Set Up Branch Protection
────────────────────────────────────────────────────────────────────────────────

  Protect your default branch so all changes go through CI.

  Go to: https://github.com/ratt-ru/surfvis/settings/rules

  Click "New ruleset" and configure:

    Name:              Protect master
    Enforcement:       Active
    Target:            Default branch
    Rules to enable:
      - Require a pull request before merging
      - Require status checks to pass (add: "Code Quality", "Tests")
      - Block force pushes

    Bypass list:
      - Add your GitHub App (created in Step 4) so it can push
        automated cab updates

  Make sure to click "Create" to save the ruleset.

────────────────────────────────────────────────────────────────────────────────
  Step 6: Make Your First Release
────────────────────────────────────────────────────────────────────────────────

  When you're ready to publish to PyPI:

    uv run tbump 0.0.1

  This will:
    1. Update version in pyproject.toml and __init__.py
    2. Regenerate cab definitions with the release version
    3. Commit, tag, and push
    4. GitHub Actions will build and publish to PyPI and ghcr.io

================================================================================
  That's it! Your CI/CD pipeline is fully configured.
================================================================================

────────────────────────────────────────────────────────────────────────────────
  Day-to-Day Development: Image Tag Workflow
────────────────────────────────────────────────────────────────────────────────

  The container image is stored in src/surfvis/_container_image.py
  as the single source of truth for cab generation and container fallback
  execution. The tag portion must stay in sync with your current context.

  When you create a feature branch:

    1. Edit src/surfvis/_container_image.py and change the tag:

         CONTAINER_IMAGE = "ghcr.io/ratt-ru/surfvis:my-feature"

    2. Commit and develop as normal — pre-commit hooks will generate cab
       definitions with the correct branch-specific image tag.

  You do NOT need to reset the tag before merging. On merge to master,
  the update-cabs workflow automatically:

    - Resets the CONTAINER_IMAGE tag to "latest"
    - Regenerates cab definitions
    - Commits _container_image.py and cab YAML files

  During releases, tbump updates the tag to the semantic version
  (e.g. 0.1.0) via its before-commit hooks.

────────────────────────────────────────────────────────────────────────────────

NOTE: Once you've completed the setup steps above, you can safely delete the
onboard command (cli/onboard.py and core/onboard.py) and remove it from
cli/__init__.py.

Add your own commands following the same pattern — define a CLI function
with type hints and a @stimela_cab decorator, and Stimela cab definitions
will be auto-generated from your CLI definitions via pre-commit hooks.

For more details, see: https://github.com/landmanbester/hip-cargo#readme
"""
    )
