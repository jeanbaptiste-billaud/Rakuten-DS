resource "docker_image" "images" {
  for_each     = local.images
  name         = each.value
  keep_locally = true
}
