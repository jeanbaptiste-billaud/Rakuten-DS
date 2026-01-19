import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
from src.utils.minio_utils import push_data, minio_bucket_connect, BucketNotFoundError

@patch("src.utils.minio_utils.Minio")
@patch.dict('os.environ', {
    "MINIO_HOST": "localhost",
    "MINIO_PORT": "9000",
    "AWS_ACCESS_KEY_ID": "minio",
    "AWS_SECRET_ACCESS_KEY": "minio123"
})
def test_minio_bucket_connect_success(mock_minio):
    """Test successful connection to an existing bucket."""
    mock_client = MagicMock()
    mock_client.bucket_exists.return_value = True
    mock_minio.return_value = mock_client

    client = minio_bucket_connect("my-bucket")
    
    assert client == mock_client
    mock_client.bucket_exists.assert_called_with("my-bucket")

@patch("src.utils.minio_utils.Minio")
def test_minio_bucket_connect_failure(mock_minio):
    """Test connection failure when bucket does not exist."""
    mock_client = MagicMock()
    mock_client.bucket_exists.return_value = False
    mock_minio.return_value = mock_client

    with pytest.raises(BucketNotFoundError):
        minio_bucket_connect("missing-bucket")

@patch("src.utils.minio_utils.minio_bucket_connect")
def test_push_data(mock_connect, tmp_path):
    """
    Test push_data logic:
    - Verifies it calls fput_object for each file.
    - Verifies the object name is relative to the source dir.
    """
    # Setup mock client
    mock_client = MagicMock()
    mock_connect.return_value = mock_client
    
    # Setup local files
    d = tmp_path / "data"
    d.mkdir()
    p1 = d / "file1.txt"
    p1.write_text("content1")
    
    subdir = d / "subdir"
    subdir.mkdir()
    p2 = subdir / "file2.txt"
    p2.write_text("content2")

    # Call function
    push_data("target-bucket", str(d))
    
    # Assertions
    assert mock_client.fput_object.call_count == 2
    
    # We verify that fput_object was called with correct relative paths
    calls = mock_client.fput_object.call_args_list
    args_list = [c[0] for c in calls] # [(bucket, object_name, file_path), ...]
    
    # file1.txt should be uploaded as 'file1.txt'
    # subdir/file2.txt should be uploaded as 'subdir/file2.txt'
    
    object_names = [args[1] for args in args_list]
    assert "file1.txt" in object_names
    assert "subdir/file2.txt" in object_names or f"subdir{os.sep}file2.txt" in object_names
