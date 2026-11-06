# Blood·alc

A private, offline blood-alcohol estimator for Android.

GitHub builds the APK for you automatically. You never need to install
Android Studio, Java, or anything else on your computer.

## Privacy

- Nothing ever leaves your phone. The app contains no network code, and a
  Content Security Policy (`connect-src 'none'`) blocks any request even if
  some code tried. The build itself checks this and refuses to produce an APK
  if it finds a network request in the app.
- Your profile and drinks are stored only in the app's private storage on
  the phone. Uninstalling the app deletes them.
- Fonts are bundled inside the app, so it works with no connection at all.
- Android shows an "internet" permission for this app. That comes from the
  standard Capacitor template and is left in place because removing it could
  not be tested here. The app never uses it, and the policy above blocks it.

## Keep this repository PRIVATE

`keystore/bloodalc.p12` is the key that signs your app. Every update must be
signed with the same key, or Android refuses to install it, and the only way
round that is uninstalling, which deletes your saved data. Keeping the key in
a private repository means only you can build updates. Do not delete it.

## Building

Every time you change anything in the repository, GitHub builds a new APK
(about 5 to 10 minutes). Watch progress under the **Actions** tab. When it
finishes, the APK appears under **Releases**.

To rebuild without changing anything: Actions -> Build APK -> Run workflow.

## Installing and updating

On the phone, open `https://github.com/<you>/<repo>/releases/latest` while
signed in to GitHub, tap the `.apk` file, and install. Updates install over
the top and keep your data.

## It is an estimate, not a measurement

Never treat any number from this app as clearance to drive.
