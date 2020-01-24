//var matrixSize = [1170,250];
//var matrixSize = [svgWidth-70,250];
var matrixSize = [svgWidth - 450, 250];


var headerCluster = d3.cluster().size(matrixSize);
var currentHeaders;
var currentHeaderDelta;

var fadedur = 400; // for fading out headers or painted areas
var headerdur = 800;
var opacityBGCircles = 0.6;

var minCellHeight = -1;

// if making headers for the first time (at vis init), then it won't animate as much to avoid confusion
function updateHeaders(firstTime) {
    var appeardur = 500;
    var delayFactor = firstTime ? 0 : 2 / 3;
    var nameDelay = firstTime ? 0 : 200;

    var nodes = d3.hierarchy(root, function (d) {
        if (!d.expandedMatrix || d.children.length == 0) return null;
        return d.children;
    })
        .sum(function (d) {
            if (!d.expandedMatrix || d.children.length == 0) return d.matches.length;
            return 0;
        })
        .sort(function (a, b) {
            // return b.value - a.value;
            return a.data.name.localeCompare(b.data.name);
        });

    headerCluster(nodes);

    var descendants = [], links = [];

    // create a linear list of a depth-first pre-traversal of the nodes
    // this is the list that will be displayed left to right as the headers
    // we use (and add more variables to) the nodes created by d3.cluster rather than the base structure 
    //   to make things more consistent across the different views

    function recursiveAdd(node) {
        if (node.data.name != 'ROOT') descendants.push(node);
        if (node.children) node.children.forEach(d => recursiveAdd(d));
    }

    recursiveAdd(nodes);

    console.log(descendants);

    // override the d3.cluster x,y with our own x,y
    // these will put the headers in a single row, with a slight vertical adjustment for deeper nodes

    var xDelta = matrixSize[0] / descendants.length;
    var yDelta = -8; // each deeper node will get offset by this vertical value

    console.log(matrixSize);

    descendants.forEach((d, i) => {
        d.x = i * xDelta + xDelta / 2;
        d.y = matrixSize[1] + yDelta * (d.depth - 1);
        if (d.children) {
            d.linkLength = (d.descendants().length - 1) * xDelta;
            links.push(d); // create a link using this node
        } else if (d.hasOwnProperty('linkLength')) delete d.linkLength;
    })

    console.log(links);

    function getLinkPath(node) {
        console.log(node);
        var height = 12;
        var s = ['M', node.x, node.y + 6, 'v', height, 'h', node.linkLength, 'v', height * -1];
        return s.join(' ');
    }

    // ----------- Links -------------
    var s = basePaintGroup
        .selectAll('path.link')
        .data(links, d => {
            console.log(d.data);
            return d.data.id;
        });

    var enter = s.enter()
        .insert('path', ':first-child')
        .attr('d', d => getLinkPath(d))
        .attr('opacity', 0)

    enter
        .transition()
        .duration(appeardur)
        .delay(headerdur * delayFactor)
        .attr('opacity', 1)

    s
        .transition()
        .duration(headerdur)
        .attr('d', d => getLinkPath(d))
        .attr('opacity', 1)

    enter
        .merge(s)
        .classed('link', true)
        .style('stroke', d => colorMapper(d.depth))

    s.exit().transition().duration(fadedur).attr('opacity', 0).remove();

    // ---------- Text ----------
    s = basePaintGroup
        .selectAll('text.names')
        .data(descendants, d => d.data.id)

    var bufx = 8, bufy = -5;

    enter = s.enter()
        .append('text')
        .attr('x', d => d.x + bufx)
        .attr('y', d => d.y + bufy)
        .attr('opacity', 0)
        .attr('transform', d => 'rotate(-60 ' + (d.x + bufx) + ',' + (d.y + bufy) + ')')
        .attr('font-size', d => (100 - 15 * (d.depth - 1)) + "%")

    enter
        .transition()
        .duration(appeardur)
        .delay(headerdur * delayFactor + nameDelay)
        .attr('opacity', 1)

    s
        .transition()
        .duration(headerdur)
        .attr('x', d => d.x + bufx)
        .attr('y', d => d.y + bufy)
        .attr('transform', d => 'rotate(-60 ' + (d.x + bufx) + ',' + (d.y + bufy) + ')')
        //.attr('font-size', d => (100 - 15 * (d.depth-1)) + "%")
        .attr('opacity', 1) // this forces interrupted animations to not have invisible text


    enter
        .merge(s)
        .classed('names', true)
        .classed('toplevel', d => d.depth == 1)
        .attr('fill', d => colorMapper(d.depth))
        .attr('font-size', d => (100 - 15 * (d.depth - 1)) + "%")
        .text(d => d.data.name)
        .on('mousedown', d => headerClicked(d))

    s.exit().transition().duration(fadedur).attr('opacity', 0).remove();

    // -------- Nodes ----------
    // (this is the old circle way)
    // s = basePaintGroup
    //   .selectAll('circle.node')
    //   .data(descendants,d => d.data.id);

    // enter = s.enter()
    //   //.insert('circle',':last-child')
    //   .append('circle')
    //   .attr('cx',d => d.x)
    //   .attr('cy',d => d.y)
    //   .attr('opacity',0)

    // enter
    //   .transition()
    //   .duration(appeardur)
    //   .delay(headerdur * delayFactor)
    //   .attr('opacity',1)

    // s
    //   .transition()
    //   .duration(headerdur)
    //   .attr('cx', function(d) {return d.x;})
    //   .attr('cy', function(d) {return d.y;})
    //   .attr('opacity',1) // this forces interrupted animations to not have invisible nodes

    // enter
    //   .merge(s)
    //   .classed('node', true)
    //   .attr('r',4)
    //   .on('mousedown',headerClicked)

    // -------- Nodes ----------
    // (this is the new plus/minus way)
    // pretty hacky way to position the plus or minus text in relation to the parent text
    // true is minus, false is plus, slightly different because the size of "+" vs "-" is different
    var bufmap = {
        "true": [bufx - 8, bufy + 12],
        "false": [bufx - 6, bufy + 15]
    }
    s = basePaintGroup
        .selectAll('text.plusminus')
        .data(descendants, d => d.data.id);

    enter = s.enter()
    //.insert('circle',':last-child')
        .append('text')
        .attr('x', d => d.x + adjust(d)[0])
        .attr('y', d => d.y + adjust(d)[1])
        .attr('transform', d => 'rotate(-60 ' + (d.x + adjust(d)[0]) + ',' + (d.y + adjust(d)[1]) + ')')
        .attr('opacity', 0)

    enter
        .transition()
        .duration(appeardur)
        .delay(headerdur * delayFactor)
        .attr('opacity', 1)

    s
        .transition()
        .duration(headerdur)
        .attr('x', d => d.x + adjust(d)[0])
        .attr('y', d => d.y + adjust(d)[1])
        .attr('transform', d => 'rotate(-60 ' + (d.x + adjust(d)[0]) + ',' + (d.y + adjust(d)[1]) + ')')
        .attr('opacity', 1) // this forces interrupted animations to not have invisible nodes

    enter
        .merge(s)
        .text(d => getPrefix(d))
        .classed('plusminus', true)
        .classed('toplevel', d => d.depth == 1)
        .attr('fill', d => colorMapper(d.depth))
        .attr('font-size', d => ((140 + (getPrefix(d) === '.' ? 20 : 0)) - 15 * (d.depth - 1)) + "%")
        .on('mousedown', headerClicked)


    s.exit().transition().duration(fadedur).attr('opacity', 0).remove();

    function getPrefix(d) {
        var pm = d.data.expandedMatrix ? "-" : "+";
        return d.depth === 3 ? "." : pm;
    }

    function adjust(d) {
        var a = [0, 0];
        switch (getPrefix(d)) {
            case '+':
                a = [bufx - 6, bufy + 15];
                break;
            case '-':
                a = [bufx - 8, bufy + 12];
                break;
            case '.':
                a = [bufx - 8, bufy + 8];
                break;
        }
        return a;
    }


    // ----------- Rectangles -----------

    var rectbufy = 10;
    var rectbufx = [20, 10]; // front and end (more at front to cover plus/minus sign)
    s = basePaintGroup
        .selectAll('rect.hoverbox')
        .data(descendants, d => d.data.id)

    enter = s.enter()
        .append('rect')
        .style('opacity', 0)
        .each(function (d) {
            var t = d3.select(this);
            var text = d3.selectAll('text.names').filter(dd => d === dd);
            var w, h;
            // really only called on one element
            text.each(function (dd) {
                var b = d3.select(this).node().getBBox();
                w = b.width;
                h = b.height;
            })
            d.data.textDim = [w, h];
            t.attr('x', d.x + bufx - rectbufx[0])
            t.attr('y', d.y + bufy - rectbufy - h * 0.8)
            t.attr('rx', (h + rectbufy * 2) / 2)
            t.attr('ry', (h + rectbufy * 2) / 2)
            t.attr('width', w + rectbufx[0] + rectbufx[1])
            t.attr('height', h + rectbufy * 2)
            t.attr('transform', 'rotate(-60 ' + (d.x + bufx) + ',' + (d.y + bufy) + ')')
        })

    enter
        .transition()
        .duration(appeardur)
        .delay(headerdur * delayFactor + nameDelay)
        .style('opacity', 1)

    s
        .transition()
        .duration(headerdur)
        .attr('x', d => d.x + bufx - rectbufx[0])
        .attr('y', d => d.y + bufy - rectbufy - d.data.textDim[1] * 0.8)
        .attr('transform', d => 'rotate(-60 ' + (d.x + bufx) + ',' + (d.y + bufy) + ')')
        .style('opacity', 1) // this forces interrupted animations to not have invisible text

    enter
        .merge(s)
        .classed('hoverbox', true)
        .style('stroke', d => colorMapper(d.depth))
        .on('mousedown', d => headerClicked(d))
        .on('mouseenter', function (d) {
            d3.select(this).classed('hovered', true)
        })
        .on('mouseleave', function (d) {
            d3.select(this).classed('hovered', false)
        })

    s.exit().remove();

    basePaintGroup.selectAll('text.plusminus').raise();
    basePaintGroup.selectAll('text.names').raise();

    currentHeaders = descendants;
    currentHeaderDelta = xDelta;

}

function headerClicked(node) {
    if (node.depth === 3) return; // don't redraw everything for child nodes, just do nothing
    node.data.expandedMatrix = !node.data.expandedMatrix;
    logEvent((node.data.expandedMatrix ? "Expanded" : "Collapsed") + " header \"" + node.data.name + "\"");
    updateHeaders(false);
    if (node.data.expandedMatrix)
        ;//matrixRows.forEach(d => d.setPaintedItems(node.data.children.map(c => c.name), true))
    else
        ;//matrixRows.forEach(d => d.setPaintedItems(node.data.children.map(c => c.name), false))
    matrixRows.forEach(d => d.drawPaintedItems());
}


var rowIdCounter = 0;

function Row(topleft, height, lifeformList, name) {
    this.topleft = topleft;
    this.height = height;

    this.name = name; // the name of this row (ie, a classification of the lifeforms we're searching through)

    this.paintTool = null;
    this.paintToolGroup = null;
    this.id = rowIdCounter++;

    // a group situated at the same location that won't have gooey effects applied to it
    // this is the "background" group where we have border lines and invisible mouse-event rectangles
    // it has to be the first element in the DOM for the overall group
    this.borderGroup = basePaintGroup.append('g')
        .datum(this)
        .attr('transform', 'translate(' + topleft.join(',') + ')')
        .style('pointer-events', 'all')

    // the layer where we draw stuff relating to the actual vis, like cells
    this.group = basePaintGroup.append('g')
        .attr('transform', 'translate(' + topleft.join(',') + ')')
        .style("filter", "url(#gooeyCodeFilter)")

    this.textGroup = basePaintGroup.append('g')
        .attr('transform', 'translate(' + topleft.join(',') + ')')


    // an invisible rectangle in the background that will receive our mouse events
    this.borderGroup
        .on('mousemove', handleRowMouseMove)
        .on('mouseout', handleRowMouseOut)
        .append('rect')
        .style('visibility', 'hidden')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', 2000)
        .attr('height', this.height)

    // this.borderGroup.append('line')
    // .attr('x1',0)
    // .attr('y1',0)
    // .attr('x2',2000)
    // .attr('y2',0)
    // .classed('rowBorder',true)
    this.borderGroup.append('line')
        .attr('x1', 0)
        .attr('y1', height)
        .attr('x2', matrixSize[0])
        .attr('y2', height)
        .classed('rowBorder', true);

    // a text object we can reposition to show the number above each cell during mouseover
    this.label = this.textGroup.append('text')
        .classed('numberLabel', true)
        .attr('visibility', 'hidden')
        .attr('pointer-events', 'none')

    // the controls for adding/removing rows	
    var dim = 18, buf = 8;
    this.borderGroup.append('image')
        .datum(this)
        .attr('x', matrixSize[0] + buf * 2)
        .attr('y', this.height / 2 - dim / 2)
        .attr('width', dim)
        .attr('height', dim)
        .attr('xlink:href', 'img/x.svg')
        .on('click', deleteRow)

    this.borderGroup.append('image')
        .datum(this)
        .attr('x', matrixSize[0] + buf * 2)
        .attr('y', this.height - dim / 2)
        .attr('width', dim)
        .attr('height', dim)
        .attr('xlink:href', 'img/plus.svg')
        .on('click', addNewRow)


    if (lifeformList)
        this.setLifeformList(lifeformList);

    //this.setLifeformList(d3.range(0,lifeforms.length));
    //this.setPaintedItems(Object.keys(groups),true);
    //this.setPaintedItems(Object.keys(allFunctionItems),true);
}

// called for the first row so that it gets auto-populated with the "all lifeforms" paint tool
Row.prototype.populateWithAll = function () {
    var tool = new PaintTool([this.topleft[0] + matrixViewOffset[0], this.topleft[1] + matrixViewOffset[1]],
        ["ROOT"], d3.range(lifeforms.length), null, colorMapper(0));
    this.setPaintTool(tool, [0, 0]);
    this.setPaintedItems(Object.keys(allFunctionItems), true);
}

Row.prototype.dispose = function () {
    this.group.remove();
    this.borderGroup.remove();
    this.textGroup.remove();
    if (this.paintToolGroup) this.paintToolGroup.remove();
    if (this.paintTool) this.paintTool.dispose();
}

Row.prototype.setLifeformList = function (list) {
    this.lifeforms = list; // lifeforms that exist in this row
    this.matches = {};

    // each group, subgroup, function gets a list of its own matches
    // we can compute this now and then never touch it again for this row, making drawing easy
    [groups, subgroups, functions].forEach((f, i) => {
        for (var key in f) this.matches[key] = {name: key, painted: false, lifeforms: [], row: this, depth: i + 1};
    })

    // fill "this.matches" with each lifeform's functions
    this.lifeforms.forEach(d => {
        lifeforms[d].Functions.forEach(func => {
            var f = func;
            // for any function, fill this.matches with the function, subgroup, and group
            // that is, go up the 'parent' tree until you hit ROOT and add it as a match to each item
            while (f.name != "ROOT") {
                var m = this.matches[f.name].lifeforms;
                // add it to the list, but only add it once
                if (m.length === 0 || m[m.length - 1] != d) m.push(d);
                f = f.parent;
            }
        })
    })

}

function handleRowMouseMove(row) {
    var m = d3.mouse(this);

    // find which header we are hovering under
    var runningX = currentHeaders[0].x;
    var i = 0;
    if (m[0] < currentHeaders[0].x) i = 0;
    else if (m[0] > currentHeaders[currentHeaders.length - 1].x) i = currentHeaders.length - 1;
    else {
        while (m[0] < runningX - currentHeaderDelta / 2 || m[0] > runningX + currentHeaderDelta / 2) {
            runningX += currentHeaderDelta;
            i++;
        }
    }
    var header = currentHeaders[i];
    var name = header.data.name;

    // toggle the header fonts to show which row is being highlighted
    //var text = basePaintGroup.selectAll('text.names')
    //	.classed('highlighted',d => d.data.name === name);

    if (row.paintTool === null && toolBeingDragged != null) {
        row.setPaintTool(toolBeingDragged);


    }

    if (toolBeingDragged && toolBeingDragged === row.paintTool) {
        row.paintItemOnClick(name, m);
        if (!header.data.expandedMatrix)
            row.setPaintedForChildren(name, true);

    }
}

function handleRowMouseOut(row) {
    basePaintGroup.selectAll('text.names')
        .classed('highlighted', false);
}

Row.prototype.setPaintedForChildren = function (parentName, value) {
    var item = this.matches[parentName];
    item.painted = value;
    var _this = this;

    allFunctionItems[parentName].children.forEach(function (d) {
        _this.setPaintedForChildren(d.name, value);
    })
}

Row.prototype.setPaintTool = function (tool, globalMousePos) {
    if (this.paintTool == null) {
        this.paintTool = tool;
        this.paintTool.row = this;

        this.setLifeformList(tool.lifeforms);//.map(d => lifeforms[d]));

        var m;
        try {
            m = d3.mouse(this.borderGroup.node());
        } catch (err) {
            m = [0, 0];
        }

        // putting the paint group as a child of 'textGroup' makes it the later in the DOM than the circles (ie, overtop)
        this.paintToolGroup = this.paintTool.paintIntoGroup(m, this.textGroup);

        this.paintToolGroup
            .transition()
            .duration(1500)
            .ease(d3.easePolyOut)
            .attr('transform', 'translate(0,0)')

        var _this = this;

        this.paintToolGroup
            .attr('pointer-events', 'all')
            .on('mousedown', function () {
                toolBeingDragged = _this.paintTool;
                _this.paintTool.group.style('opacity', 1)
            })
            .on('mouseup', function () {
            })


    }
}

Row.prototype.paintItemOnClick = function (itemName, clickPos) {
    var item = this.matches[itemName];
    //console.log('this match',item)
    //console.log('count',Object.values(this.matches).filter(d => d.painted).length);
    if (!item.painted) {
        item.painted = true;
        item.header = currentHeaders.filter(d => d.data.name === itemName)[0];

        logEvent('Painted cell "' + itemName + '" with tool "' + this.paintTool.name + '"');

        var items = [item];
        var _this = this;

        // add the item and all its children to the list of things we have to draw
        allFunctionItems[item.name].children.forEach(e => {
            items.push(_this.matches[e.name]);
            e.children.forEach(f => items.push(_this.matches[f.name]));
        })
        // remove items that have already been painted, because they will be positioned under their header properly
        // items (+ the parent item) will now contain everything that needs to be either drawn for the first time, or repositioned
        items = items.filter(d => !d.painted || d === item);

        items.forEach(d => {
            d.lifeformPercentage = d.lifeforms.length / maxLifeforms;
            //if(d.lifeforms.length === 0) d.lifeformPercentage = 1; // show a full-size hatched block for 0 matches

            if (d.header && d.painted) { // then the header has been expanded and the circle belongs under it
                d.circCenter = [d.header.x, _this.height / 2];

                // sway this circle only if it has no children
                // that is, if there are moving circles around this cell, don't also sway the 'top' cell
                d.boxBoundary = null;
                if (allFunctionItems[d.name].children.length === 0 || allFunctionItems[d.name].expandedMatrix) {
                    var myRad = getRadius(d, _this.height);
                    var boxWidth = Math.max(currentHeaderDelta, myRad * 2);
                    var left = d.header.x - boxWidth / 2 + myRad;
                    var right = d.header.x + boxWidth / 2 - myRad;
                    d.boxBoundary = {left: left, right: right, top: myRad, bottom: _this.height - myRad};
                }
            } else {
                // this is a child node that needs to be drawn around its parent in a random way
                var parent = allFunctionItems[d.name].parent;
                while (!_this.matches[parent.name].header || !_this.matches[parent.name].painted)
                    parent = parent.parent;
                var match = _this.matches[parent.name];
                // with the parent found, find a random position that won't push the child circle outside the box
                // we could put the circle anywhere in the box (width: max(currentHeaderDelta, parent's radius), height: this.height)
                // but let's cut it off so the radius of the child circle is not outside this box
                var boxWidth = Math.max(currentHeaderDelta, getRadius(match, _this.height) * 2);
                var myRad = getRadius(d, _this.height);
                var left = match.header.x - boxWidth / 2 + myRad;
                var right = match.header.x + boxWidth / 2 - myRad;

                //d.circCenter = [match.header.x, _this.height/2];
                d.circCenter = [random(left, right), random(0 + myRad, _this.height - myRad)];
                d.boxBoundary = {left: left, right: right, top: myRad, bottom: _this.height - myRad};
            }
            d.primaryCircle = d.header && d.painted; // is this the primary circle for its cell? need to know for sorting DOM elements
        })

        // rect variables
        // var len = item.lifeforms.length;
        // if(len === 0) len = maxLifeforms;
        // var rectHeight = Math.max(minCellHeight, (len / maxLifeforms) * this.height);
        // var topleft = [item.header.x - currentHeaderDelta / 2, clickPos[1] - rectHeight / 2]
        // if(topleft[1] < 0) topleft[1] = 0;
        // if(topleft[1] + rectHeight > this.height) topleft[1] = this.height - rectHeight;

        // pick only the groups that could possibly contain old items
        var s = this.group.selectAll('g').filter(d => items.includes(d))
            .data(items, d => allFunctionItems[d.name].id)

        var enter = s.enter();
        // set up the circles where they're going to be, with radius 0
        enter.append('g')
            .append('circle');
        var merge = enter.merge(s);

        var allGroups = this.group.selectAll('g');

        // merge
        // 	.selectAll('circle')
        // 	.attr('cx',d => d === item ? clickPos[0] : d.circCenter[0])
        // 	.attr('cy',d => d === item ? clickPos[1] : d.circCenter[1])
        // 	.attr('r',d => items.includes(d) ? 20 : 5);

        // we can't just use the 's' variable (which has filtered already) because, for some reason,
        // it applies to all circles. so instead, we have to re-filter here
        allGroups.filter(d => items.includes(d))
            .each(function (d) {
                d3.select(this).selectAll('*').interrupt(); // kill old transitions
                var c = d3.select(this).selectAll('circle')
                c.attr('cx', d === item ? clickPos[0] : d.circCenter[0])
                    .attr('cy', d === item ? clickPos[1] : d.circCenter[1])
                    .attr('r', 0)
                    .style('opacity', d.primaryCircle ? 1 : opacityBGCircles)
            })

        var circleFillDur = 800;
        var circleAppearDur = 800;
        var paintedCircleMoveDur = 1500;

        var clickedCircle = allGroups.filter(d => items.includes(d) && d === item);
        var otherCircles = allGroups.filter(d => items.includes(d) && d !== item);

        clickedCircle
            .selectAll('circle')
            .transition()
            .duration(circleFillDur)
            .ease(d3.easeCubicOut)
            .attr('r', d => Math.max(minCellHeight, Math.sqrt(d.lifeformPercentage) * _this.height / 2))
            .transition()
            .duration(paintedCircleMoveDur)
            .attr('cx', d => d.circCenter[0])
            .attr('cy', d => d.circCenter[1])
            .on('end', function (d) {
                if (d.boxBoundary) swayCell(d, d3.select(this));
            });

        otherCircles
            .selectAll('circle')
            .transition()
            .duration(circleAppearDur)
            .delay(circleFillDur)
            .attr('r', d => Math.max(minCellHeight, Math.sqrt(d.lifeformPercentage) * _this.height / 2))
            .on('end', function (d) {
                if (d.boxBoundary) swayCell(d, d3.select(this));
            })

        // stuff in common with all items (newly added or still there)
        merge
            .selectAll('circle')
            .classed('paintedRect', true)
            .on('mousedown', cellMouseDown)
            .on('mousemove', cellMouseMove)
            .on('mouseup', cellMouseUp)
            .on('mouseover', cellMouseOver)
            .on('mouseout', cellMouseOut)
            //.style('fill',d => d.lifeforms.length === 0 ? "url(#diagonalHatch)" : colorMapper(d.header.depth))
            .style('fill', d => d.lifeforms.length === 0 ? "#e3e3e3" : colorMapper(d.depth))

        // enter.append('g')
        // .append('circle')
        // .attr('cx',d => d === item ? clickPos[0] : d.circCenter[0])
        // .attr('cy',d => d === item ? clickPos[1] : d.circCenter[1])
        // .attr('r',0)
        // .transition()
        // .duration(800)
        // .delay(d => d === item ? 0 : 800)
        // .attr('cx',d => d.circCenter[0])
        // .attr('cy',d => d.circCenter[1])
        // .attr('r',d => Math.max(minCellHeight, Math.sqrt(d.lifeformPercentage) * _this.height / 2))
        // .on('end',function(d) { if(d.boxBoundary) swayCell(d, d3.select(this)); })


        //.style('pointer-events','all')

        sortInDom(allGroups);
    }

}

function cellMouseDown(d) {
    if (!d.primaryCircle) return;
    itemClicked = d;
}

function cellMouseMove(d) {
    if (!d.primaryCircle) return;
    // spawn a tool to be dragged, but we have to save this tool and let the global mouse events move it around
    // since we will be moving this tool outside the scope of this node
    if (itemClicked != null && toolBeingDragged == null && d.row.paintTool != null) {
        // the new tool will be the old tool's filter list, coupled with the filter from this cell
        // we don't have to worry about duplicate names and stuff, the creation of the tool takes care of that for us
        toolBeingDragged = new PaintTool(d3.mouse(this), [...d.row.paintTool.filterList, d.name], d.lifeforms, null, colorMapper(d.header.depth));
        itemClicked = null;
    }
}

function cellMouseUp(d) {
    if (!d.primaryCircle) return;
    if (toolBeingDragged == null) {
        // this is where a regular "click" event would go
        // populate the lifeforms list
        // the name for this list would be the same as if we had generated a paint tool using this cell	
        populateLifeformsList(generateColoredPaintToolName([...d.row.paintTool.filterList, d.name]), d.lifeforms);
        logEvent('Looked at lifeform list for "' + generatePaintToolName([...d.row.paintTool.filterList, d.name]) + '"')
    }
    itemClicked = null;
}

function cellMouseOver(cell) {
    if (!cell.primaryCircle) return;
    //console.log('cell is',cell);
    // highlight the name of the current cell
    //var text = basePaintGroup.selectAll('text.names')
    //	.classed('highlighted',d => d.data.name === cell.name);
    basePaintGroup.selectAll('rect.hoverbox')
        .classed('hovered', d => d.data.name === cell.name);

    // show the number of lifeforms for this cell
    // move the row's predetermined label to above the cell
    var circ = d3.select(this);
    cell.row.label
        .attr('visibility', 'visible')
        .text(cell.lifeforms.length);
    // get bounding box info for the text so we can center it
    var box = cell.row.label.node().getBBox();
    cell.row.label
        .attr('x', parseInt(circ.attr('cx')) - box.width / 2)
        .attr('y', circ.attr('cy') - circ.attr('r') - 5)

}

function cellMouseOut(cell) {
    if (!cell.primaryCircle) return;
    cell.row.label
        .attr('visibility', 'hidden')
    basePaintGroup.selectAll('rect.hoverbox')
        .classed('hovered', false);

}

Row.prototype.setPaintedItems = function (itemsNames, value) {
    itemsNames.forEach(d => {
        this.matches[d].painted = value;
    })

    this.drawPaintedItems();
}

Row.prototype.drawPaintedItems = function () {
    // if we haven't set a paint tool yet, we have nothing to draw here
    if (!this.matches)
        return;

    var _this = this;

    var headerNames = currentHeaders.map(d => d.data.name);
    //var items = Object.values(this.matches)
    //	.filter(d => d.painted && headerNames.includes(d.name));

    var items = [];
    // get all painted groups and all their children
    Object.keys(groups).filter(d => _this.matches[d].painted).forEach(d => {
        items.push(_this.matches[d]);
        allFunctionItems[d].children.forEach(e => {
            items.push(_this.matches[e.name]);
            e.children.forEach(f => items.push(_this.matches[f.name]));
        })
    });

    //console.log('items',items)

    var s = this.group.selectAll('g')
        .data(items, d => allFunctionItems[d.name].id)

    // some setup for all items; map current header positions to each element, find the rect center
    Object.values(this.matches).forEach(d => d.header = null);
    currentHeaders.forEach(d => {
        _this.matches[d.data.name].header = d;
    })

    var enter = s.enter();

    enter.merge(s).each(d => {
        d.lifeformPercentage = d.lifeforms.length / maxLifeforms;
        //if(d.lifeforms.length === 0) d.lifeformPercentage = 1; // show a full-size hatched block for 0 matches

        if (d.header && d.painted) { // then the header has been expanded and the circle belongs under it
            d.circCenter = [d.header.x, _this.height / 2];

            // sway this circle only if it has no children
            // that is, if there are moving circles around this cell, don't also sway the 'top' cell
            d.boxBoundary = null;
            //console.log('d',d);
            if (allFunctionItems[d.name].children.length === 0 || allFunctionItems[d.name].expandedMatrix) {
                var myRad = getRadius(d, _this.height);
                var boxWidth = Math.max(currentHeaderDelta, myRad * 2);
                var left = d.header.x - boxWidth / 2 + myRad;
                var right = d.header.x + boxWidth / 2 - myRad;
                d.boxBoundary = {left: left, right: right, top: myRad, bottom: _this.height - myRad};
            }
        } else {
            // this is a child node that needs to be drawn around its parent in a random way
            var parent = allFunctionItems[d.name].parent;
            while (!_this.matches[parent.name].header || !_this.matches[parent.name].painted)
                parent = parent.parent;
            var match = _this.matches[parent.name];
            // with the parent found, find a random position that won't push the child circle outside the box
            // we could put the circle anywhere in the box (width: max(currentHeaderDelta, parent's radius), height: this.height)
            // but let's cut it off so the radius of the child circle is not outside this box
            var boxWidth = Math.max(currentHeaderDelta, getRadius(match, _this.height) * 2);
            var myRad = getRadius(d, _this.height);
            var left = match.header.x - boxWidth / 2 + myRad;
            var right = match.header.x + boxWidth / 2 - myRad;

            //d.circCenter = [match.header.x, _this.height/2];
            d.circCenter = [random(left, right), random(0 + myRad, _this.height - myRad)];
            d.boxBoundary = {left: left, right: right, top: myRad, bottom: _this.height - myRad};
        }
        d.primaryCircle = d.header && d.painted; // is this the primary circle for its cell? need to know for sorting DOM elements
    })
        .selectAll('*').interrupt(); // kill old transitions


    // the new items that weren't there before
    enter = enter
        .append('g');

    enter
        .append('circle')
        .attr('cx', d => d.circCenter[0])
        .attr('cy', d => d.circCenter[1])
        .attr('r', 0)
        .style('opacity', d => d.primaryCircle ? 1 : opacityBGCircles)
        .transition()
        .duration(800)
        .delay(headerdur * 2 / 3)
        .attr('r', d => Math.max(minCellHeight, Math.sqrt(d.lifeformPercentage) * _this.height / 2))
        .on('end', function (d) {
            if (d.boxBoundary) swayCell(d, d3.select(this));
        })


    // the items that were there before
    s
        .selectAll('circle')
        .transition()
        .duration(headerdur)
        .attr('cx', d => d.circCenter[0])
        .attr('cy', d => d.circCenter[1])
        .attr('r', d => Math.max(minCellHeight, Math.sqrt(d.lifeformPercentage) * _this.height / 2))
        .style('opacity', d => d.primaryCircle ? 1 : opacityBGCircles)
        .on('end', function (d) {
            if (d.boxBoundary) swayCell(d, d3.select(this));
        })

    // stuff in common with all items (newly added or still there)
    enter
        .merge(s)
        .selectAll('circle')
        .classed('paintedRect', true)
        .on('mousedown', cellMouseDown)
        .on('mousemove', cellMouseMove)
        .on('mouseup', cellMouseUp)
        .on('mouseover', cellMouseOver)
        .on('mouseout', cellMouseOut)
        .style('fill', d => d.lifeforms.length === 0 ? "#e3e3e3" : colorMapper(d.depth))
    //.style('fill',d => d.lifeforms.length === 0 ? "url(#diagonalHatch)" : colorMapper(d.header.depth))


    // the items that are no longer there
    //s.exit().transition().duration(6000).remove(); //attr('opacity',0)//.remove();
    s.exit().selectAll('circle').transition().duration(fadedur).style('opacity', 0)
        .on('end', function () {
            s.exit().remove();
        })

    sortInDom(enter.merge(s));
}

// sort the groups (containing the circles) in the DOM
// we want all non-primary circles to be first (ie, in the background)
// then, of the primary circles, ones with higher depth need to be later (ie, foreground)
// d3's sort function sorts the selection and also reorders the DOM for us to match the new order
function sortInDom(selection) {
    selection.sort((a, b) => {
        if (a.primaryCircle === b.primaryCircle) {
            if (a.primaryCircle) return a.depth - b.depth;
            else return b.depth - a.depth;
        } else if (a.primaryCircle) return 1;
        else return -1;
    });

}

function swayCell(d, node) {
    //if(d.boxBoundary.right <= d.boxBoundary.left) console.log('oh no');
    var pos = [node.attr('cx'), node.attr('cy')];
    var myRad = node.attr('r');
    // if this cell can't animate because it fills the cell, then give up on it
    if (d.boxBoundary.right <= d.boxBoundary.left && d.boxBoundary.bottom <= d.boxBoundary.top)
        return;
    var newCenter = [random(d.boxBoundary.left, d.boxBoundary.right), random(d.boxBoundary.top, d.boxBoundary.bottom)];
    var distMoved = length(sub(pos, newCenter));
    var speed = 3000 / 15; // take 3000 ms to move 15 screen units (e.g.)

    // transition the node from wherever it is to the new position, then loop back and repeat
    node.transition().duration(speed * distMoved).ease(d3.easeLinear)//.ease(d3.easeQuadInOut)
        .attr('cx', newCenter[0])
        .attr('cy', newCenter[1])
        .on('end', function (d) {
            swayCell(d, node);
        });
}

function getRadius(match, rowHeight) {
    return Math.sqrt(match.lifeformPercentage) * rowHeight / 2;
}

Row.prototype.moveRow = function (newTopLeft, animateTime, easeFunction) {
    this.topleft = newTopLeft;
    if (!easeFunction) easeFunction = d3.easeCubic;
    this.borderGroup
        .transition()
        .duration(animateTime)
        .ease(easeFunction)
        .attr('transform', 'translate(' + newTopLeft.join(',') + ')')

    this.group
        .transition()
        .duration(animateTime)
        .ease(easeFunction)
        .attr('transform', 'translate(' + newTopLeft.join(',') + ')')

    this.textGroup
        .attr('transform', 'translate(' + newTopLeft.join(',') + ')')
}

Row.prototype.fadeOut = function (animateTime) {
    this.borderGroup
        .transition()
        .duration(animateTime)
        .style('opacity', 0)

    this.group
        .transition()
        .duration(animateTime)
        .style('opacity', 0)
}

