provider "infisical" {
  host = var.infisical_host
  auth = {
    token = var.bootstrap_token
  }
}

resource "infisical_project" "rakuten" {
  name                       = var.project_name
  slug                       = var.project_slug
  type                       = "secret-manager"
  should_create_default_envs = true
}

locals {
  folder_paths = toset(distinct(flatten([
    for identity_name, identity in var.identities : concat(
      identity.path != null && identity.path != "" ? [identity.path] : [],
      identity.paths
    )
  ])))

  folder_parts = {
    for path in local.folder_paths : path => compact(split("/", trim(path, "/")))
  }

  all_folders = merge([
    for path, parts in local.folder_parts : {
      for index, part in parts : join("/", slice(parts, 0, index + 1)) => {
        parent = index == 0 ? "/" : "/${join("/", slice(parts, 0, index))}"
        name   = part
        depth  = index + 1
      }
    }
  ]...)

  root_folders = {
    for key, folder in local.all_folders : key => folder
    if folder.depth == 1
  }

  child_folders = {
    for key, folder in local.all_folders : key => folder
    if folder.depth > 1
  }
}

resource "infisical_secret_folder" "root_folders" {
  for_each = local.root_folders

  project_id       = infisical_project.rakuten.id
  environment_slug = var.environment_slug
  folder_path      = each.value.parent
  name             = each.value.name
  description      = "Managed by Terraform for Rakuten DS V2."
}

resource "infisical_secret_folder" "child_folders" {
  for_each = local.child_folders

  project_id       = infisical_project.rakuten.id
  environment_slug = var.environment_slug
  folder_path      = each.value.parent
  name             = each.value.name
  description      = "Managed by Terraform for Rakuten DS V2."

  depends_on = [
    infisical_secret_folder.root_folders
  ]
}

resource "infisical_identity" "identities" {
  for_each = var.identities

  org_id = var.organization_id
  name   = each.key
  role   = "member"

  metadata = [
    {
      key   = "managed_by"
      value = "terraform"
    },
    {
      key   = "project"
      value = var.project_slug
    }
  ]
}

resource "infisical_identity_universal_auth" "universal_auth" {
  for_each = {
    for name, identity in var.identities : name => identity
    if name == var.terraform_identity_name
  }

  identity_id                 = infisical_identity.identities[each.key].id
  access_token_ttl            = 3600
  access_token_max_ttl        = 3600
  access_token_num_uses_limit = 0

  client_secret_trusted_ips = [
    {
      ip_address = "0.0.0.0/0"
    }
  ]

  access_token_trusted_ips = [
    {
      ip_address = "0.0.0.0/0"
    }
  ]
}

resource "infisical_identity_universal_auth_client_secret" "client_secrets" {
  for_each = {
    for name, identity in var.identities : name => identity
    if name == var.terraform_identity_name
  }

  identity_id = infisical_identity.identities[each.key].id
  description = "Terraform provider Universal Auth client secret."

  depends_on = [
    infisical_identity_universal_auth.universal_auth
  ]
}

resource "infisical_project_identity" "project_identities" {
  for_each = var.identities

  project_id  = infisical_project.rakuten.id
  identity_id = infisical_identity.identities[each.key].id

  roles = [
    {
      role_slug = "no-access"
    }
  ]
}

resource "infisical_project_identity_specific_privilege" "identity_privileges" {
  for_each = {
    for item in flatten([
      for identity_name, identity in var.identities : [
        for path in concat(
          identity.path != null && identity.path != "" ? [identity.path] : [],
          identity.paths
          ) : {
          key          = "${identity_name}:${path}"
          identity_id  = infisical_project_identity.project_identities[identity_name].identity_id
          project_slug = infisical_project.rakuten.slug
          slug         = identity_name
          path         = path
          actions      = identity.actions
        }
      ]
    ]) : item.key => item
  }

  project_slug = each.value.project_slug
  identity_id  = each.value.identity_id
  slug         = each.value.slug

  permissions_v2 = [
    {
      action  = each.value.actions
      subject = "secrets"
      conditions = jsonencode({
        environment = {
          "$eq" = var.environment_slug
        }
        secretPath = {
          "$eq" = each.value.path
        }
      })
    }
  ]

  depends_on = [
    infisical_secret_folder.child_folders
  ]
}
