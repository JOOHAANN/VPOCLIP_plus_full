#!/usr/bin/env bash
set -Eeuo pipefail

# Run this only after the ETRI account/session in /home/youhan/.etri-curl.conf
# has been refreshed.  It keeps the archives so the download is recoverable.
COOKIE_CONFIG="/home/youhan/.etri-curl.conf"
BASE_URL="https://epretx.etri.re.kr"
DATA_PAGE_URL="$BASE_URL/dataFileList?lang=en&id=65"
DOWNLOAD_DIR="/home/youhan/ws/ETRI-Activity3D-Skeleton-archives"
SKELETON_ROOT="/home/youhan/ws/ETRI-Activity3D-Skeleton"
mkdir -p "$DOWNLOAD_DIR"

for id in 327 328; do
  if [[ "$id" == 327 ]]; then
    archive="$DOWNLOAD_DIR/Skeleton(P001-P050).zip"
  else
    archive="$DOWNLOAD_DIR/Skeleton(P051-P100).zip"
  fi
  partial="$archive.part"
  if [[ -s "$archive" ]]; then
    echo "ETRI_SKELETON_ARCHIVE_EXISTS id=$id file=$archive"
    continue
  fi

  response="$(curl --config "$COOKIE_CONFIG" --user-agent 'Mozilla/5.0' \
    --referer "$DATA_PAGE_URL" --header 'X-Requested-With: XMLHttpRequest' -sS -X POST \
    -d "id=$id" "$BASE_URL/updateDataFileCount")"
  if [[ "$response" == *"로그인해주세요"* || "$response" != *"dataFile"* ]]; then
    echo "ETRI_SKELETON_DOWNLOAD_BLOCKED id=$id: refresh the authenticated ETRI session in $COOKIE_CONFIG" >&2
    exit 4
  fi

  echo "ETRI_SKELETON_DOWNLOAD_START id=$id"
  curl --config "$COOKIE_CONFIG" --fail --location --show-error \
    --retry 10 --retry-delay 10 --retry-max-time 0 --retry-all-errors \
    --remove-on-error --user-agent 'Mozilla/5.0' \
    --referer "$DATA_PAGE_URL" \
    --output "$partial" "$BASE_URL/download?id=$id"
  unzip -tq "$partial"
  mkdir -p "$SKELETON_ROOT"
  unzip -q "$partial" -d "$SKELETON_ROOT"
  mv -- "$partial" "$archive"
  echo "ETRI_SKELETON_DOWNLOAD_COMPLETE id=$id file=$archive"
done

echo "ETRI_SKELETON_ARCHIVES_READY $DOWNLOAD_DIR"
