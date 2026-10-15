// Android only for now: iOS is not built (see README).
module.exports = {
  dependency: {
    platforms: {
      ios: null,
      android: {
        packageImportPath: 'import dev.kiln.reactnative.KilnPackage;',
        packageInstance: 'new KilnPackage()',
      },
    },
  },
};
