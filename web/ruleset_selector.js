'use strict';
(function(root){
  const engines={'bun-native-v3-items':root.QQT,'bun-native-v2-face':root.QQTBunNativeV2Face,'bun-native-v2':root.QQTBunNativeV2,'stock-window-v1':root.QQTStockWindowV1};
  for(const api of Object.values(engines))root.QQTR3Inputs.installSim(api);
  root.QQTRulesets={current:'bun-native-v3-items',select(version){
    if(!Object.hasOwn(engines,version))throw Error('ruleset');
    this.current=version;root.QQT=engines[version];return root.QQT;
  }};
})(globalThis);
