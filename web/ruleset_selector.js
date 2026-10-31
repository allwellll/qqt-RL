'use strict';
(function(root){const v2=root.QQT;root.QQTR3Inputs.installSim(v2);root.QQTR3Inputs.installSim(root.QQTStockWindowV1);root.QQTRulesets={current:'bun-native-v2',select(version){if(!['bun-native-v2','stock-window-v1'].includes(version))throw Error('ruleset');this.current=version;root.QQT=version==='bun-native-v2'?v2:root.QQTStockWindowV1;return root.QQT;}};})(globalThis);
