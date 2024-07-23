# Versioning
version_full ?= $(shell $(MAKE) --silent version-full)
version_small ?= $(shell $(MAKE) --silent version)
# DevTunnel configuration
tunnel_name := call-center-ai-$(shell hostname | sed 's/[^a-zA-Z0-9]//g' | tr '[:upper:]' '[:lower:]')
tunnel_url ?= $(shell res=$$(devtunnel show $(tunnel_name) | grep -o 'http[s]*://[^"]*' | xargs) && echo $${res%/})
# App location
cognitive_communication_location := westeurope
default_location := swedencentral
functionapp_location := swedencentral
openai_location := swedencentral
search_location := francecentral
# Sanitize variables
name_sanitized := $(shell echo $(name) | tr '[:upper:]' '[:lower:]')
# App configuration
bot_phone_number ?= $(shell cat config.yaml | yq '.communication_services.phone_number')
event_subscription_name ?= $(shell echo '$(name_sanitized)-$(bot_phone_number)' | tr -dc '[:alnum:]-')
twilio_phone_number ?= $(shell cat config.yaml | yq '.sms.twilio.phone_number')
# Bicep outputs
app_url ?= $(shell az deployment sub show --name $(name_sanitized) | yq '.properties.outputs["appUrl"].value')
blob_storage_public_name ?= $(shell az deployment sub show --name $(name_sanitized) | yq '.properties.outputs["blobStoragePublicName"].value')
communication_id ?= $(shell az deployment sub show --name $(name_sanitized) | yq '.properties.outputs["communicationId"].value')
function_app_name ?= $(shell az deployment sub show --name $(name_sanitized) | yq '.properties.outputs["functionAppName"].value')
log_analytics_workspace_customer_id ?= $(shell az deployment sub show --name $(name_sanitized) | yq '.properties.outputs["logAnalyticsWorkspaceName"].value')

version:
	@bash ./cicd/version/version.sh -g . -c

version-full:
	@bash ./cicd/version/version.sh -g . -c -m

brew:
	@echo "➡️ Installing yq..."
	brew install yq

	@echo "➡️ Installing Azure CLI..."
	brew install azure-cli

	@echo "➡️ Installing Azure Functions Core Tools..."
	brew tap azure/functions && brew install azure-functions-core-tools@4

	@echo "➡️ Installing Azure Dev tunnels..."
	curl -sL https://aka.ms/DevTunnelCliInstall | bash

	@echo "➡️ Installing Syft..."
	brew install syft

	@echo "➡️ Installing SOPS..."
	brew install sops

	@echo "➡️ Installing age..."
	brew install age

	@echo "➡️ Installing Twilio CLI..."
	brew tap twilio/brew && brew install twilio

install:
	@echo "➡️ Installing pip-tools..."
	python3 -m pip install pip-tools

	@echo "➡️ Syncing dependencies..."
	pip-sync --pip-args "--no-deps" requirements-dev.txt

upgrade:
	@echo "➡️ Updating Git submodules..."
	git submodule update --init --recursive

	@echo "➡️ Upgrading pip..."
	python3 -m pip install --upgrade pip setuptools wheel

	@echo "➡️ Upgrading pip-tools..."
	python3 -m pip install --upgrade pip-tools

	@echo "➡️ Compiling app requirements..."
	pip-compile \
		--output-file requirements.txt \
		pyproject.toml

	@echo "➡️ Compiling dev requirements..."
	pip-compile \
		--extra dev \
		--output-file requirements-dev.txt \
		pyproject.toml

	@echo "➡️ Upgrading Bicep CLI..."
	az bicep upgrade

test:
	@$(MAKE) static-test
	@$(MAKE) unit-test

static-test:
	@echo "➡️ Test generic formatter (Black)..."
	python3 -m black --check .

	@echo "➡️ Test import formatter (isort)..."
	python3 -m isort --jobs -1 --check .

	@echo "➡️ Test dependencies issues (deptry)..."
	python3 -m deptry .

	@echo "➡️ Test code smells (Pylint)..."
	python3 -m pylint .

	@echo "➡️ Test types (Pyright)..."
	python3 -m pyright .

unit-test:
	@echo "➡️ Unit tests (Pytest)..."
	PUBLIC_DOMAIN=dummy pytest \
		--html=test-reports/$(version_full).html \
		--junit-xml=test-reports/$(version_full).xml \
		tests/*.py

lint:
	@echo "➡️ Fix with generic formatter (Black)..."
	python3 -m black .

	@echo "➡️ Fix with import formatter (isort)..."
	python3 -m isort --jobs -1 .

tunnel:
	@echo "➡️ Creating tunnel..."
	devtunnel show $(tunnel_name) || devtunnel create $(tunnel_name) --allow-anonymous --expiration 1d

	@echo "➡️ Creating port forwarding..."
	devtunnel port show $(tunnel_name) --port-number 8080 || devtunnel port create $(tunnel_name) --port-number 8080

	@echo "➡️ Starting tunnel..."
	devtunnel host $(tunnel_name)

dev:
	VERSION=$(version_full) PUBLIC_DOMAIN=$(tunnel_url) func start

deploy:
	@echo "👀 Current subscription:"
	@az account show --query "{subscriptionId:id, subscriptionName:name, tenantId:tenantId}" --output table

	@echo "🛠️ Deploying resources..."
	az deployment sub create \
		--location $(default_location) \
		--parameters \
			'cognitiveCommunicationLocation=$(cognitive_communication_location)' \
			'functionappLocation=$(functionapp_location)' \
			'instance=$(name)' \
			'openaiLocation=$(openai_location)' \
			'searchLocation=$(search_location)' \
			'version=$(version_full)' \
		--template-file bicep/main.bicep \
	 	--name $(name_sanitized)

	@echo "💤 Wait 10 secs for output to be available..."
	sleep 10

	@echo "🛠️ Deploying Function App..."
	func azure functionapp publish $(function_app_name) \
		--build local \
		--build-native-deps \
		--python

	@echo "🚀 Call Center AI is running on $(app_url)"

	@$(MAKE) post-deploy name=$(name_sanitized)

post-deploy:
	@$(MAKE) copy-resources \
		name=$(blob_storage_public_name)

	@$(MAKE) twilio-register \
		endpoint=$(app_url)

	@$(MAKE) logs name=$(name_sanitized)

destroy:
	@echo "🧐 Are you sure you want to delete? Type 'delete now $(name_sanitized)' to confirm."
	@read -r confirm && [ "$$confirm" = "delete now $(name_sanitized)" ] || (echo "Confirmation failed. Aborting."; exit 1)

	@echo "❗️ Deleting RG..."
	az group delete --name $(name_sanitized) --yes --no-wait

	@echo "❗️ Deleting deployment..."
	az deployment sub delete --name $(name_sanitized)

logs:
	func azure functionapp logstream $(function_app_name) \
		--browser

twilio-register:
	@echo "⚙️ Registering Twilio webhook..."
	twilio phone-numbers:update $(twilio_phone_number) \
		--sms-url $(endpoint)/twilio/sms

copy-resources:
	@echo "📦 Copying resources to Azure storage account..."
	az storage blob upload-batch \
		--account-name $(name_sanitized) \
		--destination '$$web' \
		--no-progress \
		--output none \
		--overwrite \
		--source resources

watch-call:
	@echo "👀 Watching status of $(phone_number)..."
	while true; do \
		clear; \
		curl -s "$(endpoint)/call?phone_number=%2B$(phone_number)" | yq --prettyPrint '.[0] | {"phone_number": .phone_number, "claim": .claim, "reminders": .reminders}'; \
		sleep 3; \
	done

sbom:
	@echo "🔍 Generating SBOM..."
	syft scan \
		--source-version $(version_full)  \
		--output spdx-json=./sbom-reports/$(version_full).json \
		.
