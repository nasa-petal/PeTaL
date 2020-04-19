var baseSVG, baseSelectGroup, basePaintGroup;
// an array of objects containing data about each lifeform, taken from the json file
// var lifeforms;

// maps that contain all the data about our functions
// each element in the map is keyed by its name
// each result is a map itself with these values:
// - name: the name of the item (duplicates the key for easy querying if you only have the object)
// - parent: a link to its immediate parent group or subgroup (or to 'root' if a group)
// - children: an array of links to its children subgroups or functions (or empty array if a function)
// - matches: an array of indices (into the lifeforms array) for lifeforms that contain this item
// var root;

// the max # of lifeforms any group, subgroup, or function has

var gaussianBlur;
var filter;
var uidiv;
var buttons;
var tip;

var matrixRows = [];
var matrixRowHeight = 150;

var matrixViewOffset = [20, 0];

var colorMapper = d3.scaleSequential(d3.interpolateMagma).domain([-3, 5]); // purple to orange scheme, the original
//var globalColorArray = ['#b73779','#e75263','#fc8961','#DB7A16']
// var colorMapper = function(v) {
// return globalColorArray[v];
// }

// global mouse variables
var toolBeingDragged = null;
var itemClicked = null;

// lifeform list div
var listHeaderDiv, listDiv;


class Tool {

    constructor() {
        this.svgWidth = 1900;
        this.matrixSize = [this.svgWidth - 450, 250];
        this.svgHeight = this.matrixSize[1] + 50 + 4 * matrixRowHeight + 50;
        this.maxLifeforms = 0;

        let that = this;

        d3.json("static/js/hierarchy.json", function (error, data) {
            if (error) {
                console.log("error loading json:", error);
            }

            // an array of objects containing data about each lifeform, taken from the json file
            // that.lifeforms = lf;
            that.lifeforms = [];
            // that.lifeformsTest = lf;

            // console.log(lf);

            // lf.forEach(d => {
            //     // delete d.functions;
            //     delete d.level0;
            //     delete d.level1;
            //     delete d.level2;
            //     delete d.permalink;
            //     delete d.post_id;
            //     delete d.post_title;
            // });

            // console.log(lf);

            // parseData();

            that.root = data;
            that.root.level = 0;

            let idCounter = 0;
            const resetExpanded = function (d) {
                d.expandedCircle = false;
                d.expandedMatrix = false;
                d.id = idCounter++;
                // console.log(d);
                if (d.children) {
                    d.children.forEach(e => {
                        if (typeof e === 'object') {
                            e.parent = d;
                            resetExpanded(e);
                        }
                    });
                }
            };

            resetExpanded(that.root);
            that.root.expandedMatrix = true;
            that.root.expandedCircle = true;

            that.groups = {};
            that.subgroups = {};
            that.functions = {};

            let lifeformsCounter = [];
            that.root.children.forEach(d => {
                that.groups[d.name] = d;
                that.groups[d.name].level = 1;
                d.children.forEach(e => {
                    that.subgroups[e.name] = e;
                    that.subgroups[e.name].level = 2;
                    e.children.forEach(f => {
                        that.functions[f.name] = f;
                        that.functions[f.name].level = 3;
                        lifeformsCounter.push(...f.children);
                        f.children.forEach(g => {
                            let lifeform = that.lifeforms.find(h => h.post_id === g);
                            if (lifeform) {
                                lifeform.functions.push({
                                    level0: d.name,
                                    level1: e.name,
                                    level2: f.name
                                });
                            } else {
                                that.lifeforms.push({
                                    'post_id': g, 
                                    'functions': [{
                                        level0: d.name,
                                        level1: e.name,
                                        level2: f.name
                                    }]
                                });
                            }
                        })
                    });

                });
            });
            // that.lifeforms = uniq(that.lifeforms);
            lifeformsCounter = uniq(lifeformsCounter);
            // console.log(lifeformsCounter);

            that.nOfLifeforms = lifeformsCounter.length;

            const visitChildrenDFS = function (node, func) {
                if (typeof node !== "undefined" && node.children && node.level < 3) {
                    node.matches = node.children.reduce((accumulator, currentValue) => {
                        return accumulator.concat(visitChildrenDFS(currentValue, func));
                    }, []);
                } else {
                    node.matches = node.children;
                }
                node.matches = uniq(node.matches);
                return node.matches;
            };

            visitChildrenDFS(that.root);

            // console.log("groups", that.groups);
            // console.log("subgroups", that.subgroups);
            // console.log("functions", that.functions);

            that.allFunctionItems = {};

            // add everything to 'allFunctionItems'
            // do we really need to split it into groups, subgroups, functions?
            [that.groups, that.subgroups, that.functions].forEach(f => {
                for (var key in f) {
                    that.allFunctionItems[key] = f[key];
                }
            });

            // console.log("allFunctionItems", that.allFunctionItems);

            // full_post_example.json how many lifeforms are in each group
            // for(let key in that.groups) {
            // 	console.log(that.groups[key].name, that.groups[key].matches.length);
            // }

            // figure out the max # of lifeforms any group, subgroup, or function has
            // we only need to check groups since everything lower cannot have more than its parent
            for (let key in that.groups) {
                if (that.groups[key].matches.length > that.maxLifeforms) {
                    that.maxLifeforms = that.groups[key].matches.length;
                }
            }

            // console.log("maxLifeforms", that.maxLifeforms);

            // uidiv = d3.select('body').append('div');
            // .style('margin','1em')

            baseSVG = d3.select('body').append('div')
            // .style('float','left')
                .style('width', '100%')
                .append("svg")
                .attr("id", "mysvg")
                // .style('float', 'left')
                //.attr("width",900)
                .attr('width', "100%")
                .attr("height", that.svgHeight)
                .attr("viewBox", "0 0 " + that.svgWidth + ' ' + (that.svgHeight))
                .on('mousedown', globalMouseDown)
                .on('mousemove', globalMouseMove)
                .on('mouseup', globalMouseUp);

            var defs = baseSVG.append("defs");
            filter = defs.append("filter").attr("id", "gooeyCodeFilter");
            gaussianBlur = filter.append("feGaussianBlur")
                .attr("in", "SourceGraphic")
                .attr("stdDeviation", "8")
                //to fix safari: http://stackoverflow.com/questions/24295043/svg-gaussian-blur-in-safari-unexpectedly-lightens-image
                .attr("color-interpolation-filters", "sRGB")
                .attr("result", "blur");
            filter.append("feColorMatrix")
                .attr("in", "blur")
                .attr("mode", "matrix")
                .attr("values", "1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 19 -9")
                .attr("result", "gooey");
            filter.append("feBlend")
                .attr("in", "SourceGraphic")
                .attr("in2", "gooey");

            var pattern = defs.append('pattern')
                .attr('id', 'diagonalHatch')
                .attr('width', 20)

                .attr('height', 20)
                .attr('patternTransform', 'rotate(45 0 0)')
                .attr('patternUnits', 'userSpaceOnUse');
            pattern
                .append('line')
                .attr('x1', 0)
                .attr('y1', 0)
                .attr('x2', 0)
                .attr('y2', 20)
                .style('stroke', '#aaa')
                .style('stroke-width', 3);
            // pattern
            // .append('line')
            // .attr('x1',0)
            // .attr('y1',0)
            // .attr('x2',20)
            // .attr('y2',0)
            // .style('stroke','#aaa')
            // .style('stroke-width',3)

            baseSelectGroup = baseSVG.append('g').attr('id', 'selectGroup').style("filter", "url(#gooeyCodeFilter)");
            //basePaintGroup = baseSVG.append('g').attr('id', 'paintGroup').attr("transform","translate(660,0)");
            basePaintGroup = baseSVG.append('g').attr('id', 'paintGroup').attr("transform", "translate(" + matrixViewOffset.join(',') + ")");

            tip = d3.tip()
                .attr('class', 'd3-tip')
                //.offset([-10, 0])
                //.offset(function(d) { return [d.r,0] })
                .html(function (d) {
                    var name = d.data.name;
                    if (name === 'ROOT') {
                        name = "All Lifeforms";
                    }
                    var a = ['<span class="active">' + name + '</span> [' + d.data.matches.length + ']'];
                    var temp = d;
                    while (temp.parent != null && temp.parent.data.name !== 'ROOT') {
                        temp = temp.parent;
                        a.push(temp.data.name);
                    }
                    a.reverse();
                    return a.join('<br>');
                    //return "blah";
                });

            baseSVG.call(tip);

            createUI();

            //updateCircles();

            that.view = new PaintView(that);

            that.view.updateHeaders(that.lifeforms, that.root, true);

            // add a line above all rows that does not belong to any row
            // this line will have a "plus" icon on it, to add a row above all other rows, and can never be deleted
            var topline = basePaintGroup.append('g')
                .datum(null)
                .attr('transform', 'translate(' + [0, that.matrixSize[1] + 50].join(',') + ')');

            topline.append('line')
                .attr('x1', 0)
                .attr('y1', 0)
                .attr('x2', that.matrixSize[0])
                .attr('y2', 0)
                .classed('rowBorder', true);

            var dim = 18, buf = 8; // same values that are in paintView.js Row constructor
            topline.append('image')
                .attr('x', that.matrixSize[0] + buf * 2)
                .attr('y', 0 - dim / 2)
                .attr('width', dim)
                .attr('height', dim)
                .attr('xlink:href', 'static/img/plus.svg')
                .on('click', (e) => {
                    return that.addNewRow(that)
                });

            let row = new Row(that, that.view, [0, that.matrixSize[1] + 50], matrixRowHeight, null, "test");
            matrixRows.push(row);
            row.populateWithAll(that.nOfLifeforms);

            that.addNewRow(matrixRows[matrixRows.length - 1]);
            that.addNewRow(matrixRows[matrixRows.length - 1]);
            that.addNewRow(matrixRows[matrixRows.length - 1]);

            createGuide();
            createLifeformList();

            // baseSVG.append('circle').attrs({
            // cx: 500,
            // cy: 500,
            // fill: "green",
            // r: 50
            // })

            // logEvent("Opened tool page (index.html)");

            d3.json("static/js/strategies.json", function (error, strategies) {
                if (error) {
                    console.log("error loading json:", error);
                }

                strategies.forEach(d => {
                    let lifeform = that.lifeforms.find(e => e.post_id === d.post_id);
                    lifeform.post_title = d.post_title;
                    lifeform.permalink = d.permalink;
                });
            });
        });
    }
}

function createGuide() {
    var div = d3.select('body').append('div')
        .classed('guideTop', true)
        .append('div')
        .classed('guideList', true);
    var s = `<ul><li>
                Angled text expands into subgroups into functions
            </li>
            <li>
                Find co-existing functions by dragging and brushing a circle from the first row across the next one
            </li>
            <li>
                Click a circle and a list of suggestions with multiple functions appears below - each one takes you to an Ask Nature strategy page  
            </li></ul>`;
    div.html(s);
}

// create the div that shows the lifeform list
function createLifeformList() {
    var div = d3.select('body').append('div')
        .classed('lifeformTop', true);

    listHeaderDiv = div.append('div')
        .classed('lifeformHeader', true)
        .html(wrapColor('Group, Sub-group and Functional Co-occurrence', colorMapper(0)));

    listDiv = div.append('div')
        .classed('lifeformList', true)
}

function wrapColor(text, color) {
    return '<span style="color:' + color + '">' + text + '</span>';
}

function createUI() {
    buttons = [
        {name: "Gooey", value: true},
        {name: "Movement", value: true},
        {name: "Gaussian", value: 6},
        {name: "feBlend", value: true},
        {name: "Blur Effect", value: "19 -9"}
    ];

    // uidiv.append("div")
    // .selectAll(".buttons")
    // .data(buttons).enter().append("input")
    // .attrs({
    // type: "button",
    // value: function(d){return d.name + ": " + d.value}
    // })
    // .on("click", function(d){
    // clickButton(d,d3.select(this));
    // });
}

function clickButton(setting, button) {
    if (setting.name == "Gooey") {
        setting.value = !setting.value;
        baseSelectGroup.style('filter', setting.value ? "url(#gooeyCodeFilter)" : null);
    } else if (setting.name == "Movement") {
        setting.value = !setting.value;
        var c = baseSelectGroup.selectAll('circle');
        if (!setting.value) {
            c.interrupt(); // cancel existing animations
            c.attr('cx', d => d.x).attr('cy', d => d.y); // set positions to proper place
        } else {
            c.each(function (d) {
                console.log(d);
                swayNode(d, d3.select(this));
            }); // start up animations again
        }
    } else if (setting.name == "Gaussian") {
        var v = [1, 3, 6, 10];
        var i = (v.indexOf(setting.value) + 1) % v.length;
        setting.value = v[i];

        gaussianBlur.attr('stdDeviation', setting.value);

    } else if (setting.name == "feBlend") {
        setting.value = !setting.value;
        if (setting.value) {
            filter.append("feBlend")
                .attr("in", "SourceGraphic")
                .attr("in2", "gooey");
        } else {
            filter.selectAll("feBlend").remove();
        }
    } else if (setting.name == "Blur Effect") {
        var v = ["19 -9", "12 -9", "12 -3"];
        var i = (v.indexOf(setting.value) + 1) % v.length;
        setting.value = v[i];

        filter.selectAll('feColorMatrix')
            .attr("values", "1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 " + setting.value)
    }
    button.attr('value', d => d.name + ": " + d.value)
}

Tool.prototype.getSetting = function (name) {
    var match = buttons.filter(d => d.name == name)
    return match.length == 1 ? match[0].value : null;
};

// parse the "Function" item in the json to create a branching tree of interlinked groups, subgroups, functions
// also, change the "Function" item from a string into an array of our "function" objects
// Tool.prototype.parseData = function () {
//     var maps = [groups, subgroups, functions];
//     root = {
//         "name": "ROOT", "parent": null, "children": [], "matches": [], "id": 0,
//         "expandedCircle": false, "expandedMatrix": true
//     };
//
//     var id = 1;
//     lifeforms.forEach((d, lifeformIndex) => {
//         root.matches.push(lifeformIndex);
//         var newfuncs = []; // we'll change the function string to an array of objects
//         var funcs = d.Functions.split('|');
//         funcs.forEach(f => {
//             var split = f.split('>');
//             // ignore all entries that don't have group > subgroup > function format
//             if (split.length === 3) {
//                 var items = [];
//                 // pull the items from the groups, or create them if they don't exist
//                 for (var i = 0; i < split.length; i++) {
//                     var map = maps[i];
//                     if (!map.hasOwnProperty(split[i]))
//                         map[split[i]] = {
//                             "name": split[i],
//                             "parent": null,
//                             "children": [],
//                             "matches": [],
//                             "id": id++,
//                             "expandedCircle": false,
//                             "expandedMatrix": false
//                         };
//                     items.push(map[split[i]]);
//                 }
//                 // items now contains 3 items [group, subgroup, function], let's tie them together for this lifeform
//                 items.forEach((item, i) => {
//                     if (i > 0)
//                         item.parent = items[i - 1];
//                     if (i < 2 && !item.children.includes(items[i + 1]))
//                         item.children.push(items[i + 1]);
//                     if (!item.matches.includes(lifeformIndex))
//                         item.matches.push(lifeformIndex);
//                 });
//                 newfuncs.push(items[2]); // save the function item to put back in this lifeform
//             }
//         });
//
//         d.Functions = newfuncs; // change the "Functions" item from a string to an array of our 'function' objects
//     });
//
//     for (var key in groups) {
//         root.children.push(groups[key]);
//         groups[key].parent = root;
//     }
//
//     // add everything to 'allFunctionItems'
//     // do we really need to split it into groups, subgroups, functions?
//     [groups, subgroups, functions].forEach(f => {
//         for (var key in f) allFunctionItems[key] = f[key];
//     });
//
// };

function globalMouseDown() {
    // prevents a bunch of weird highlighting of text/elements which breaks our mouse interaction
    d3.event.preventDefault();
}

function globalMouseMove() {
    if (toolBeingDragged != null) {
        var m = d3.mouse(this);
        toolBeingDragged.setPosition(m)

    }
}

function globalMouseUp() {
    if (toolBeingDragged != null) {
        var m = d3.mouse(this);
        toolBeingDragged.stopDragging();
        toolBeingDragged = null;
    }
}

function populateLifeformsList(headerName, lifeformList) {
    // listHeaderDiv.html(headerName + " (" + lifeformList.length + ")");
    // var s = "<ul>";
    // lifeformList.forEach((d,i) => {
    // 	var life = lifeforms[d];
    // 	s += '<li><a class="natureLink" href="' + life.Permalink + '" target="_blank">' + life.Title + "</a></li>"
    // 	//s += '<li>' + life.Title + ' (<a href="' + life.Permalink + '">link</a>)</li>';
    // })
    // s += "</ul>";
    // listDiv.html(s);

    // $('a.natureLink').mousedown(function(e) {
    // 	// left or middle click
    // 	if(e.which === 1 || e.which === 2) {
    // 		console.log('e is',e);
    // 		logEvent("clicked");
    // 	}
    // })

    listHeaderDiv.html(headerName + " (" + lifeformList.length + ")");
    listDiv.selectAll('*').remove();
    var ul = listDiv.append('ul');
    ul.selectAll('li')
        .data(lifeformList)
        .enter()
        .append('li')
        .each(function (d, i) {
            var link = d3.select(this).append('a');
            console.log(t.lifeforms[d]);
            link.attr('href', t.lifeforms[d].permalink)
                .attr('target', '_blank')
                .html(t.lifeforms[d].post_title)
                // .on('mousedown', function () {
                //     if (d3.event.which === 1 || d3.event.which === 2)
                //         // logEvent('Clicked link to "' + t.lifeforms[d].post_title + '"');
                // })
        })
};

Tool.prototype.addNewRow = function (aboveRow) {
    // console.log(aboveRow);
    // console.log(this);

    var index = -1;
    if (aboveRow)
        index = matrixRows.indexOf(aboveRow);
    var r = new Row(this, this.view, [0, this.matrixSize[1] + 50 + (index + 1) * matrixRowHeight], matrixRowHeight, null, "test");
    r.borderGroup.style('opacity', 0); // hide it and animate it appearing after other rows are done moving

    matrixRows.splice(index + 1, 0, r);

    // if the row we are adding is the only one in the whole vis, fill it
    if (matrixRows.length === 1)
        r.populateWithAll(that.lifeforms.length);

    var delay = 0; // animate the new row appearing immediately, unless rows have to move
    // if it wasn't the last entry in matrix, we have to shift the others down
    if (index + 1 < matrixRows.length - 1) {
        for (var i = index + 2; i < matrixRows.length; i++)
            matrixRows[i].moveRow([0, this.matrixSize[1] + 50 + i * matrixRowHeight], 700)
        delay = 400;
    }

    d3.timeout(function () {
        r.borderGroup.transition().duration(600).style('opacity', 1);
    }, delay);

    this.svgHeight = this.matrixSize[1] + 50 + matrixRows.length * matrixRowHeight + 50;
    baseSVG.attr('height', this.svgHeight);
    // baseSVG.attr('width', this.svgWidth);
    baseSVG.attr('viewBox', '0 0 ' + this.svgWidth + ' ' + this.svgHeight);

    // logEvent("Added new row");
};

Tool.prototype.deleteRow = function (row) {
    let that = row.tool;
    // console.log(that.matrixSize);

    var index = matrixRows.indexOf(row);

    // logEvent("Deleted " + (row.paintTool === null ? "blank row" : ("row \"" + row.paintTool.name + '"')));

    // fade out the current row
    var fadetime = 600;
    row.fadeOut(fadetime);

    d3.timeout(function () {
        row.dispose();
        matrixRows.splice(index, 1); // remove the row

        that.svgHeight = that.matrixSize[1] + 50 + matrixRows.length * matrixRowHeight + 50;
        baseSVG.attr('height', that.svgHeight);
        // baseSVG.attr('width', that.svgWidth);
        baseSVG.attr('viewBox', '0 0 ' + that.svgWidth + ' ' + that.svgHeight);

        // move the rows up that are below this one, if any
        if (index < matrixRows.length) {
            for (var i = index; i < matrixRows.length; i++)
                matrixRows[i].moveRow([0, that.matrixSize[1] + 50 + i * matrixRowHeight], 700, d3.easeBounce);
        }

        // if we deleted our only row, make a new blank one
        if (matrixRows.length === 0) {
            addNewRow(null);
        }

    }, fadetime);

};

// function logEvent(str) {
//     console.log("Event:", str);
//     let url = '/logging/' + str;
//     $.post(url, {}, function () {
//     });
// }

