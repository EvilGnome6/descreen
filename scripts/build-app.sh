#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/app/build/Descreen.app"
SRC="$ROOT/app/Descreen"

if [ ! -x "$SRC/Resources/engine/descreen-engine" ]; then
  "$ROOT/scripts/sync-engine.sh"
fi

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

xcrun swiftc -parse-as-library \
  "$SRC/AppKitMain.swift" \
  "$SRC/Models.swift" \
  "$SRC/EngineClient.swift" \
  -o "$APP/Contents/MacOS/Descreen" \
  -framework AppKit \
  -framework UniformTypeIdentifiers \
  -target arm64-apple-macosx14.0 \
  -O

cat > "$APP/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key>
	<string>en</string>
	<key>CFBundleExecutable</key>
	<string>Descreen</string>
	<key>CFBundleIdentifier</key>
	<string>com.mike.descreen</string>
	<key>CFBundleName</key>
	<string>Descreen</string>
	<key>CFBundleDisplayName</key>
	<string>Descreen</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleShortVersionString</key>
	<string>0.1.0</string>
	<key>CFBundleVersion</key>
	<string>1</string>
	<key>LSMinimumSystemVersion</key>
	<string>14.0</string>
	<key>NSHighResolutionCapable</key>
	<true/>
	<key>NSPrincipalClass</key>
	<string>NSApplication</string>
	<key>CFBundleDocumentTypes</key>
	<array>
		<dict>
			<key>CFBundleTypeName</key>
			<string>JPEG Image</string>
			<key>CFBundleTypeRole</key>
			<string>Viewer</string>
			<key>LSHandlerRank</key>
			<string>Default</string>
			<key>CFBundleTypeExtensions</key>
			<array>
				<string>jpg</string>
				<string>jpeg</string>
				<string>JPG</string>
				<string>JPEG</string>
			</array>
		</dict>
	</array>
</dict>
</plist>
PLIST

cp -R "$SRC/Resources/engine" "$APP/Contents/Resources/engine"
chmod +x "$APP/Contents/Resources/engine/descreen-engine" "$APP/Contents/MacOS/Descreen"
printf 'APPL????' > "$APP/Contents/PkgInfo"
codesign --force --deep -s - "$APP" >/dev/null

echo "Built $APP"
echo "Open with: open \"$APP\""
