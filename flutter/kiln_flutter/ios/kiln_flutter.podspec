Pod::Spec.new do |s|
  s.name             = 'kiln_flutter'
  s.version          = '0.2.0'
  s.summary          = 'On-device LLM runtime for Flutter. Honest about what it runs on.'
  s.description      = 'Dart FFI bindings for the kiln C core (llama.cpp based). Static xcframework, Metal on Apple silicon.'
  s.homepage         = 'https://github.com/RaihanulHaque/kiln'
  s.license          = { :type => 'MIT', :file => '../LICENSE' }
  s.author           = { 'Raihanul Haque' => 'raihanulhaque007@gmail.com' }
  s.source           = { :path => '.' }
  s.platform         = :ios, '15.0'
  s.swift_version    = '5.0'

  # A dummy source so CocoaPods builds a target; the real code is the vendored static library.
  s.source_files     = 'Classes/**/*'
  s.vendored_frameworks = 'Frameworks/kiln.xcframework'
  s.frameworks       = 'Metal', 'MetalKit', 'Foundation', 'Accelerate'
  s.libraries        = 'c++'

  # Dart loads symbols with DynamicLibrary.process(), so nothing references them at link
  # time; force the whole archive in and keep the exported symbols.
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'OTHER_LDFLAGS' => '-force_load "$(PODS_TARGET_SRCROOT)/Frameworks/kiln.xcframework/ios-arm64/libkiln-merged.a"',
    'STRIP_STYLE' => 'non-global',
    'DEAD_CODE_STRIPPING' => 'NO',
  }
  s.user_target_xcconfig = { 'STRIP_STYLE' => 'non-global' }
end
