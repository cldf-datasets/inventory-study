function compare() {

  var gcode = document.getElementById('gcode').value.split(' // ')[1];
  var varieties, i, j, language, languageB;
  var textsA = [];
  var textsB = [];
  var text, sound;
  var score, score2;
  var common, inA, inB;
  var common2, inA2, inB2;
  var cls;
  var k;
  if (gcode in DATA) {
    varieties = DATA[gcode];
    for (i=0; i<varieties.length; i++) {
      language = varieties[i];
      text = '<h3>'+language['Name']+' ('+language['Dataset']+', '+language['ID']+')</h3>';
      text += '<h4>CLTS vs. Graphemes</h4>';
      text += '<div class="bordered">';

      for (sound in language['CLTS']) {
	cls = '';
	if (sound != language['CLTS'][sound]){
	  cls = ' unequal';
	}
        text += '<span class="sound'+cls+'">'+sound+'</span><span class="grapheme">'+language['CLTS'][sound]+'</span><span class="empty"> </span>';
      }
      text += '</div>';
      text += '<h4>Graphemes</h4>';
      text += '<div class="bordered">';
      text += '<span class="grapheme">'+language['Sounds'].join('</span> <span class="grapheme">')+'</span></div>';
      textsA.push(text);

      text = '';
      for (j=0; j<varieties.length; j++) {
        if (i < j){
          languageB = varieties[j];
          common = [];
          inA = [];
          inB = [];
          for (sound in language['CLTS']){
            if (sound in languageB['CLTS']){
              common.push(sound);
            }
            else {
              inA.push(sound);
            }
          }
          for (sound in languageB['CLTS']){
            if (common.indexOf(sound) == -1){
              inB.push(sound);
            }
          }
          common2 = [];
          inA2 = [];
          inB2 = [];
          for (k=0; k<language['Sounds'].length; k++){
            if (languageB['Sounds'].indexOf(language['Sounds'][k]) == -1) {
              inA2.push(language['Sounds'][k]);
            }
            else {
              common2.push(language['Sounds'][k]);
            }
          }
          for (k=0; k<languageB['Sounds'].length; k++){
            if (language['Sounds'].indexOf(languageB['Sounds'][k]) == -1){
              inB2.push(languageB['Sounds'][k]);
            }
          }
	  inA.sort();
	  inB.sort();
	  inA2.sort();
	  inB2.sort();
	  common.sort();
	  common2.sort();

    score = Math.round(100*common.length / (inA.length+inB.length+common.length));
	  score2 = Math.round(100*common2.length / (inA2.length+inB2.length+common2.length));
          text = '<h3>Compare '+language['Name']+' ('+language['ID']+', '+language['Dataset']+') vs. '+languageB['Name']+' ('+language['ID']+', '+languageB['Dataset']+'): '+score2+' / '+score+'</h3>';

          text += '<table class="comparetable">';
          text += '<tr><th>Name</th><td>'+language['Name']+'</td><td>'+languageB['Name']+'</td></tr>';
          text += '<tr><th>ID</th><td>'+language['ID']+'</td><td>'+languageB['ID']+'</td></tr>';
          text += '<tr><th>Dataset</th><td>'+language['Dataset']+'</td><td>'+languageB['Dataset']+'</td></tr>';
          text += '<tr><th>Source</th><td>'+language['Source']+'</td><td>'+languageB['Source']+'</td></tr>';
          text += '</table>';

          text += '<table class="comparetable"><tr><th>'+language['ID']+
            ' ('+language['Dataset']+')</th><th>Common</th><th>'+
            languageB['ID']+' ('+languageB['Dataset']+')</th></tr>';

          if (inA2.length > 0){
            text += '<tr><td title="'+inA2.length+'"><span class="grapheme">'+inA2.join('</span> <span class="grapheme">')+'</span></td>';
          }
          else {
            text += '<tr><td></td>';
          }
          if (common2.length > 0){
            text += '<td title="'+common2.length+'"><span class="grapheme">'+common2.join('</span> <span class="grapheme">')+'</span></td>';
          }
          else {
            text += '<td></td>';
          }
          if (inB2.length > 0){
            text += '<td title="'+inB2.length+'"><span class="grapheme">'+inB2.join('</span> <span class="grapheme">')+'</span></td>';
          }
          else {
            text += '<td></td>';
          }
          text += '</tr>';


          if (inA.length > 0){
            text += '<tr><td title="'+inA.length+'"><span class="sound">'+inA.join('</span> <span class="sound">')+'</span></td>';
          }
          else {
            text += '<tr><td></td>';
          }
          if (common.length > 0){
            text += '<td title="'+common.length+'"><span class="sound">'+common.join('</span> <span class="sound">')+'</span></td>';
          }
          else {
            text += '<td></td>';
          }
          if (inB.length > 0){
            text += '<td title="'+inB.length+'"><span class="sound">'+inB.join('</span> <span class="sound">')+'</span></td>';
          }
          else {
            text += '<td></td>';
          }
          text += '</tr></table>';

          textsB.push(text);
        }
      }
    }
    var out = document.getElementById('results');
    out.innerHTML = '';
    out.innerHTML = textsA.join(' ');
    out.innerHTML += textsB.join(' ');
  }
  for (i=0; sound=document.getElementsByClassName('sound')[i]; i++){
    sound.title = DATA['bipa-'+sound.innerHTML];
  }
}

function prepare(){
  var key;
  var languages = [];
  var i;
  
  for (key in DATA){
    if (key.indexOf('bipa-') == -1){
      for (i=0; i<DATA[key].length; i++){
        if (languages.indexOf(DATA[key][i]['Name']+' ('+DATA[key].length+') // '+key) == -1){
          languages.push(DATA[key][i]['Name']+' ('+DATA[key].length+') // '+key);
        }
      }
    }
  }
  languages.sort();
  var ipt = document.getElementById('gcode');
  var asp = new Awesomplete(ipt);
  asp.list = languages;
}
